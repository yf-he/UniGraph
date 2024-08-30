import os
import argparse
import random
import yaml
import logging
from functools import partial
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from transformers import AutoTokenizer

import dgl

import torch
import torch.nn as nn
from torch import optim as optim



logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True


def get_current_lr(optimizer):
    return optimizer.state_dict()["param_groups"][0]["lr"]

def mask_edge(graph, mask_prob):
    E = graph.num_edges()

    mask_rates = torch.ones(E) * mask_prob
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx


def drop_edge(graph, drop_rate, return_edges=False):
    if drop_rate <= 0:
        return graph

    graph = graph.remove_self_loop()

    n_node = graph.num_nodes()
    edge_mask = mask_edge(graph, drop_rate)
    src, dst = graph.edges()

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]

    ng = dgl.graph((nsrc, ndst), num_nodes=n_node)
    ng = ng.add_self_loop()

    return ng

def build_args():
    parser = argparse.ArgumentParser(description="unigraph")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--run_entity", type=str, default="xxx")
    parser.add_argument("--datasets_name", nargs='+', type=str, default=['arxiv', 'products'])
    parser.add_argument("--eval_datasets_name", nargs='+', type=str, default=['arxiv', 'products'])
    parser.add_argument("--lm_type", type=str, default="microsoft/deberta-base")
    parser.add_argument("--gnn_type", type=str, default="")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=4)

    parser.add_argument("--lr", type=float, default=2e-5,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="weight decay")
    parser.add_argument("--lr_f", type=float, default=0.01)
    parser.add_argument("--weight_decay_f", type=float, default=0.0)
    parser.add_argument("--negative_slope", type=float, default=0.1, help="the negative slope of leaky relu for GAT")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument("--mask_rate", type=float, default=0.75)
    parser.add_argument("--cut_off", type=int, default=64)
    parser.add_argument("--num_roots", type=int, default=100)
    parser.add_argument("--length", type=int, default=4)
    parser.add_argument("--process_mode", type=str, default="TA")
    parser.add_argument("--activation", type=str, default="prelu")
    parser.add_argument("--norm", type=str, default="layernorm")
    parser.add_argument("--drop_edge_rate", type=float, default=0.0)
    # parser.add_argument("--loss_type", type=str, default="sce")

    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lp_epochs", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=1000)

    parser.add_argument("--pooler", type=str, default="mean")

    parser.add_argument("--lam", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.995)
    parser.add_argument("--delayed_ema_epoch", type=int, default=0)
    
    parser.add_argument("--task", type=str, default="nc")

    parser.add_argument("--incontext_eval", action="store_true", default=False)
    parser.add_argument("--eval_num_label", type=int, default=5)
    parser.add_argument("--eval_num_support", type=int, default=3)
    parser.add_argument("--eval_num_query", type=int, default=3)

    parser.add_argument("--linear_prob_seeds", type=int, nargs="+", default=[i for i in range(1)])

    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--checkpoint_path", type=str, default="")
    args = parser.parse_args()
    return args


def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")
    else:
        return nn.Identity


def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
    opt_lower = opt.lower()

    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    return optimizer


# -------------------
def mask_edge(graph, mask_prob):
    E = graph.num_edges()

    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx


def drop_edge(graph, drop_rate, return_edges=False):
    if drop_rate <= 0:
        return graph

    n_node = graph.num_nodes()
    edge_mask = mask_edge(graph, drop_rate)
    src = graph.edges()[0]
    dst = graph.edges()[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]

    ng = dgl.graph((nsrc, ndst), num_nodes=n_node)
    ng = ng.add_self_loop()

    dsrc = src[~edge_mask]
    ddst = dst[~edge_mask]

    if return_edges:
        return ng, (dsrc, ddst)
    return ng


def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    if args.dataset not in configs:
        logging.info("Best args not found")
        return args

    logging.info("Using best configs")
    configs = configs[args.dataset]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    print("------ Use best configs ------")
    return args




class NormLayer(nn.Module):
    def __init__(self, hidden_dim, norm_type):
        super().__init__()
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "graphnorm":
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        else:
            raise NotImplementedError
        
    def forward(self, graph, x):
        tensor = x
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        batch_list = graph.batch_num_nodes
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias

# os util ==============================
def init_random_state(seed=0):
    # Libraries using GPU should be imported after specifying GPU-ID
    import torch
    import random
    # import dgl
    # dgl.seed(seed)
    # dgl.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dir_of_file(f_name):
    return os.path.dirname(f_name) + '/'


def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    import errno
    if os.path.exists(path): return
    # print(path)
    # path = path.replace('\ ',' ')
    # print(path)
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise


def init_path(dir_or_file):
    path = get_dir_of_file(dir_or_file)
    if not os.path.exists(path):
        mkdir_p(path)
    return dir_or_file


# dgl utils ===============================
 
def sample_nodes(g, seed_nodes, fanout_list):
    # seed_nodes = th.tensor(seed_nodes).to(g.device) if isinstance(seed_nodes, int) else seed_nodes
    induced_nodes = {0: (cur_nodes := seed_nodes.view(-1))}
    init_random_state(0)
    for l, fanout in enumerate(fanout_list):
        frontier = dgl.sampling.sample_neighbors(g, cur_nodes, fanout)
        cur_nodes = frontier.edges()[0].unique()
        induced_nodes[l + 1] = cur_nodes
    sampled_nodes = th.cat(list(induced_nodes.values())).unique()
    return sampled_nodes, induced_nodes


def get_edge_set(g: dgl.DGLGraph):
    """graph_edge_to list of (row_id, col_id) tuple
    """

    return set(map(tuple, np.column_stack([_.cpu().numpy() for _ in g.edges()]).tolist()))


def edge_set_to_inds(edge_set):
    """ Unpack edge set to row_ids, col_ids"""
    return list(map(list, zip(*edge_set)))

def pool(pooling):
    if pooling == "mean":
        pooler = AvgPooling()
    elif pooling == "max":
        pooler = MaxPooling()
    elif pooling == "sum":
        pooler = SumPooling()
    else:
        raise NotImplementedError
    return pooler


### Evaluator for graph classification
class Evaluator:
    def __init__(self, name):
        self.name = name

        # meta_info = pd.read_csv(os.path.join(os.path.dirname(__file__), 'master.csv'), index_col=0, keep_default_na=False)
        # if not self.name in meta_info:
        #     print(self.name)
        #     error_mssg = 'Invalid dataset name {}.\n'.format(self.name)
        #     error_mssg += 'Available datasets are as follows:\n'
        #     error_mssg += '\n'.join(meta_info.keys())
        #     raise ValueError(error_mssg)

        self.num_tasks = 1 if name == 'ogbg-molhiv' else 128
        #int(meta_info[self.name]['num tasks'])
        self.eval_metric = 'rocauc'
        #meta_info[self.name]['eval metric']


    def _parse_and_check_input(self, input_dict):
        if self.eval_metric == 'rocauc' or self.eval_metric == 'ap' or self.eval_metric == 'rmse' or self.eval_metric == 'acc':
            if not 'y_true' in input_dict:
                raise RuntimeError('Missing key of y_true')
            if not 'y_pred' in input_dict:
                raise RuntimeError('Missing key of y_pred')

            y_true, y_pred = input_dict['y_true'], input_dict['y_pred']

            '''
                y_true: numpy ndarray or torch tensor of shape (num_graphs, num_tasks)
                y_pred: numpy ndarray or torch tensor of shape (num_graphs, num_tasks)
            '''

            # converting to torch.Tensor to numpy on cpu
            if torch is not None and isinstance(y_true, torch.Tensor):
                y_true = y_true.detach().cpu().numpy()

            if torch is not None and isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.detach().cpu().numpy()


            ## check type
            if not isinstance(y_true, np.ndarray):
                raise RuntimeError('Arguments to Evaluator need to be either numpy ndarray or torch tensor')

            if not y_true.shape == y_pred.shape:
                raise RuntimeError('Shape of y_true and y_pred must be the same')

            if not y_true.ndim == 2:
                raise RuntimeError('y_true and y_pred mush to 2-dim arrray, {}-dim array given'.format(y_true.ndim))

            if not y_true.shape[1] == self.num_tasks:
                raise RuntimeError('Number of tasks for {} should be {} but {} given'.format(self.name, self.num_tasks, y_true.shape[1]))

            return y_true, y_pred

        elif self.eval_metric == 'F1':
            if not 'seq_ref' in input_dict:
                raise RuntimeError('Missing key of seq_ref')
            if not 'seq_pred' in input_dict:
                raise RuntimeError('Missing key of seq_pred')

            seq_ref, seq_pred = input_dict['seq_ref'], input_dict['seq_pred']

            if not isinstance(seq_ref, list):
                raise RuntimeError('seq_ref must be of type list')

            if not isinstance(seq_pred, list):
                raise RuntimeError('seq_pred must be of type list')

            if len(seq_ref) != len(seq_pred):
                raise RuntimeError('Length of seq_true and seq_pred should be the same')

            return seq_ref, seq_pred

        else:
            raise ValueError('Undefined eval metric %s ' % (self.eval_metric))


    def eval(self, input_dict):

        if self.eval_metric == 'rocauc':
            y_true, y_pred = self._parse_and_check_input(input_dict)
            return self._eval_rocauc(y_true, y_pred)
        if self.eval_metric == 'ap':
            y_true, y_pred = self._parse_and_check_input(input_dict)
            return self._eval_ap(y_true, y_pred)
        elif self.eval_metric == 'rmse':
            y_true, y_pred = self._parse_and_check_input(input_dict)
            return self._eval_rmse(y_true, y_pred)
        elif self.eval_metric == 'acc':
            y_true, y_pred = self._parse_and_check_input(input_dict)
            return self._eval_acc(y_true, y_pred)
        elif self.eval_metric == 'F1':
            seq_ref, seq_pred = self._parse_and_check_input(input_dict)
            return self._eval_F1(seq_ref, seq_pred)
        else:
            raise ValueError('Undefined eval metric %s ' % (self.eval_metric))

    @property
    def expected_input_format(self):
        desc = '==== Expected input format of Evaluator for {}\n'.format(self.name)
        if self.eval_metric == 'rocauc' or self.eval_metric == 'ap':
            desc += '{\'y_true\': y_true, \'y_pred\': y_pred}\n'
            desc += '- y_true: numpy ndarray or torch tensor of shape (num_graphs, num_tasks)\n'
            desc += '- y_pred: numpy ndarray or torch tensor of shape (num_graphs, num_tasks)\n'
            desc += 'where y_pred stores score values (for computing AUC score),\n'
            desc += 'num_task is {}, and '.format(self.num_tasks)
            desc += 'each row corresponds to one graph.\n'
            desc += 'nan values in y_true are ignored during evaluation.\n'
        elif self.eval_metric == 'rmse':
            desc += '{\'y_true\': y_true, \'y_pred\': y_pred}\n'
            desc += '- y_true: numpy ndarray or torch tensor of shape (num_graphs, num_tasks)\n'
            desc += '- y_pred: numpy ndarray or torch tensor of shape (num_graphs, num_tasks)\n'
            desc += 'where num_task is {}, and '.format(self.num_tasks)
            desc += 'each row corresponds to one graph.\n'
            desc += 'nan values in y_true are ignored during evaluation.\n'
        elif self.eval_metric == 'acc':
            desc += '{\'y_true\': y_true, \'y_pred\': y_pred}\n'
            desc += '- y_true: numpy ndarray or torch tensor of shape (num_nodes num_tasks)\n'
            desc += '- y_pred: numpy ndarray or torch tensor of shape (num_nodes num_tasks)\n'
            desc += 'where y_pred stores predicted class label (integer),\n'
            desc += 'num_task is {}, and '.format(self.num_tasks)
            desc += 'each row corresponds to one graph.\n'
        elif self.eval_metric == 'F1':
            desc += '{\'seq_ref\': seq_ref, \'seq_pred\': seq_pred}\n'
            desc += '- seq_ref: a list of lists of strings\n'
            desc += '- seq_pred: a list of lists of strings\n'
            desc += 'where seq_ref stores the reference sequences of sub-tokens, and\n'
            desc += 'seq_pred stores the predicted sequences of sub-tokens.\n'
        else:
            raise ValueError('Undefined eval metric %s ' % (self.eval_metric))

        return desc

    @property
    def expected_output_format(self):
        desc = '==== Expected output format of Evaluator for {}\n'.format(self.name)
        if self.eval_metric == 'rocauc':
            desc += '{\'rocauc\': rocauc}\n'
            desc += '- rocauc (float): ROC-AUC score averaged across {} task(s)\n'.format(self.num_tasks)
        elif self.eval_metric == 'ap':
            desc += '{\'ap\': ap}\n'
            desc += '- ap (float): Average Precision (AP) score averaged across {} task(s)\n'.format(self.num_tasks)
        elif self.eval_metric == 'rmse':
            desc += '{\'rmse\': rmse}\n'
            desc += '- rmse (float): root mean squared error averaged across {} task(s)\n'.format(self.num_tasks)
        elif self.eval_metric == 'acc':
            desc += '{\'acc\': acc}\n'
            desc += '- acc (float): Accuracy score averaged across {} task(s)\n'.format(self.num_tasks)
        elif self.eval_metric == 'F1':
            desc += '{\'F1\': F1}\n'
            desc += '- F1 (float): F1 score averaged over samples.\n'
        else:
            raise ValueError('Undefined eval metric %s ' % (self.eval_metric))

        return desc

    def _eval_rocauc(self, y_true, y_pred):
        '''
            compute ROC-AUC averaged across tasks
        '''

        rocauc_list = []

        for i in range(y_true.shape[1]):
            #AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
                # ignore nan values
                is_labeled = y_true[:,i] == y_true[:,i]
                rocauc_list.append(roc_auc_score(y_true[is_labeled,i], y_pred[is_labeled,i]))

        if len(rocauc_list) == 0:
            raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

        return {'rocauc': sum(rocauc_list)/len(rocauc_list)}


    def _eval_ap(self, y_true, y_pred):
        '''
            compute Average Precision (AP) averaged across tasks
        '''

        ap_list = []

        for i in range(y_true.shape[1]):
            #AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
                # ignore nan values
                is_labeled = y_true[:,i] == y_true[:,i]
                ap = average_precision_score(y_true[is_labeled,i], y_pred[is_labeled,i])

                ap_list.append(ap)

        if len(ap_list) == 0:
            raise RuntimeError('No positively labeled data available. Cannot compute Average Precision.')

        return {'ap': sum(ap_list)/len(ap_list)}

    def _eval_rmse(self, y_true, y_pred):
        '''
            compute RMSE score averaged across tasks
        '''
        rmse_list = []

        for i in range(y_true.shape[1]):
            # ignore nan values
            is_labeled = y_true[:,i] == y_true[:,i]
            rmse_list.append(np.sqrt(((y_true[is_labeled,i] - y_pred[is_labeled,i])**2).mean()))

        return {'rmse': sum(rmse_list)/len(rmse_list)}

    def _eval_acc(self, y_true, y_pred):
        acc_list = []

        for i in range(y_true.shape[1]):
            is_labeled = y_true[:,i] == y_true[:,i]
            correct = y_true[is_labeled,i] == y_pred[is_labeled,i]
            acc_list.append(float(np.sum(correct))/len(correct))

        return {'acc': sum(acc_list)/len(acc_list)}

    def _eval_F1(self, seq_ref, seq_pred):
        # '''
        #     compute F1 score averaged over samples
        # '''

        precision_list = []
        recall_list = []
        f1_list = []

        for l, p in zip(seq_ref, seq_pred):
            label = set(l)
            prediction = set(p)
            true_positive = len(label.intersection(prediction))
            false_positive = len(prediction - label)
            false_negative = len(label - prediction)

            if true_positive + false_positive > 0:
                precision = true_positive / (true_positive + false_positive)
            else:
                precision = 0

            if true_positive + false_negative > 0:
                recall = true_positive / (true_positive + false_negative)
            else:
                recall = 0
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0

            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

        return {'precision': np.average(precision_list),
                'recall': np.average(recall_list),
                'F1': np.average(f1_list)}