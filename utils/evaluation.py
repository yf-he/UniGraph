import copy
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim as optim
import numpy as np
import random

from torch.utils.data import DataLoader
from utils.data_util import load_dataset
from utils.functions import set_random_seed

from data import Mol


def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)

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

def node_classification_evaluation(graph, x, labels, split_idx, num_classes, lr_f, weight_decay_f, max_epoch_f, device, mute=False):
    in_feat = x.shape[1]
    encoder = LogisticRegression(in_feat, num_classes)

    num_finetune_params = [p.numel() for p in encoder.parameters() if  p.requires_grad]
    if not mute:
        print(f"num parameters for finetuning: {sum(num_finetune_params)}")
    
    encoder.to(device)
    optimizer_f = create_optimizer("adam", encoder, lr_f, weight_decay_f)
    #num_nodes = graph.num_nodes()
    final_acc, estp_acc, best_val_acc = linear_probing_for_transductive_node_classification(encoder, graph, x.shape[0], x, labels, split_idx, optimizer_f, max_epoch_f, device, mute)
    return final_acc, estp_acc, best_val_acc

def edge_classification_evaluation(graph, x, node_pairs, labels, split_idx, num_classes, lr_f, weight_decay_f, max_epoch_f, device, mute=False):
    in_feat = x.shape[1]*2
    encoder = LogisticRegression(in_feat, num_classes)

    #labels = torch.cat([labels['train'], labels['valid'], labels['test']], dim=0)
    # train_cat_x = torch.cat([x[node_pairs['train'][:, 0]], x[node_pairs['train'][:, 1]]], dim=1)
    # valid_cat_x = torch.cat([x[node_pairs['valid'][:, 0]], x[node_pairs['valid'][:, 1]]], dim=1)
    # test_cat_x = torch.cat([x[node_pairs['test'][:, 0]], x[node_pairs['test'][:, 1]]], dim=1)
    # x = torch.cat([train_cat_x, valid_cat_x, test_cat_x], dim=0)
    x = torch.cat([x[node_pairs[:, 0]], x[node_pairs[:, 1]]], dim=1)
    #split_idx = {'train': torch.arange(0, train_cat_x.shape[0]), 'valid': torch.arange(train_cat_x.shape[0], train_cat_x.shape[0]+valid_cat_x.shape[0]), 'test': torch.arange(train_cat_x.shape[0]+valid_cat_x.shape[0], train_cat_x.shape[0]+valid_cat_x.shape[0]+test_cat_x.shape[0])}
    print(f"split_idx: {split_idx['train'].shape}, {split_idx['valid'].shape}, {split_idx['test'].shape}")

    num_finetune_params = [p.numel() for p in encoder.parameters() if  p.requires_grad]
    if not mute:
        print(f"num parameters for finetuning: {sum(num_finetune_params)}")
    
    encoder.to(device)
    optimizer_f = create_optimizer("adam", encoder, lr_f, weight_decay_f)
    final_acc, estp_acc, best_val_acc = linear_probing_for_transductive_node_classification(encoder, graph, x.shape[0], x, labels, split_idx, optimizer_f, max_epoch_f, device, mute)
    return final_acc, estp_acc, best_val_acc

def graph_classification_evaluation(x, labels, split_idx, num_classes, lr_f, weight_decay_f, max_epoch_f, evaluator, device, mute=False):
    in_feat = x.shape[1]
    encoder = LogisticRegression(in_feat, num_classes)

    num_finetune_params = [p.numel() for p in encoder.parameters() if  p.requires_grad]
    if not mute:
        print(f"num parameters for finetuning: {sum(num_finetune_params)}")
    
    encoder.to(device)
    optimizer_f = create_optimizer("adam", encoder, lr_f, weight_decay_f)
    #num_nodes = graph.num_nodes()
    final_acc, estp_acc, best_val_acc = linear_probing_for_graph_classification(encoder, x.shape[0], x, labels, split_idx, optimizer_f, max_epoch_f, evaluator, device, mute)
    return final_acc, estp_acc, best_val_acc


def linear_probing_for_transductive_node_classification(model, graph, num_nodes, feat, labels, split_idx, optimizer, max_epoch, device, mute=False):
    criterion = torch.nn.CrossEntropyLoss()

    graph = graph.to(device)
    x = feat.to(device)
    labels = labels.to(device)

    train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    if not torch.is_tensor(train_idx):
        train_idx = torch.as_tensor(train_idx)
        val_idx = torch.as_tensor(val_idx)
        test_idx = torch.as_tensor(test_idx)

    #num_nodes = graph.num_nodes()
    train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx, True).to(device)
    val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx, True).to(device)
    test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True).to(device)
    # train_mask = graph.ndata["train_mask"]
    # val_mask = graph.ndata["val_mask"]
    # test_mask = graph.ndata["test_mask"]
    # labels = graph.ndata["label"]

    best_val_acc = 0
    best_val_epoch = 0
    best_model = None

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    for epoch in epoch_iter:
        model.train()
        out = model(graph, x)
        loss = criterion(out[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
        optimizer.step()

        with torch.no_grad():
            model.eval()
            pred = model(graph, x)
            val_acc = accuracy(pred[val_mask], labels[val_mask])
            val_loss = criterion(pred[val_mask], labels[val_mask])
            test_acc = accuracy(pred[test_mask], labels[test_mask])
            test_loss = criterion(pred[test_mask], labels[test_mask])
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_model = copy.deepcopy(model)

        if not mute:
            epoch_iter.set_description(f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, val_acc:{val_acc}, test_loss:{test_loss.item(): .4f}, test_acc:{test_acc: .4f}")

    best_model.eval()
    with torch.no_grad():
        pred = best_model(graph, x)
        estp_test_acc = accuracy(pred[test_mask], labels[test_mask])
    if mute:
        print(f"# IGNORE: --- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")
    else:
        print(f"--- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")

    # (final_acc, es_acc, best_acc)
    return test_acc, estp_test_acc, best_val_acc

def linear_probing_for_graph_classification(model, num_nodes, feat, labels, split_idx, optimizer, max_epoch, evaluator, device, mute=False):
    criterion = torch.nn.BCEWithLogitsLoss()

    #graph = graph.to(device)
    x = feat.to(device)
    labels = labels.to(device)
    #model = model.to(device)

    train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    if not torch.is_tensor(train_idx):
        train_idx = torch.as_tensor(train_idx)
        val_idx = torch.as_tensor(val_idx)
        test_idx = torch.as_tensor(test_idx)
    print(train_idx.shape, val_idx.shape, test_idx.shape)


    #num_nodes = graph.num_nodes()
    train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx, True).to(device)
    val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx, True).to(device)
    test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True).to(device)
    # train_mask = graph.ndata["train_mask"]
    # val_mask = graph.ndata["val_mask"]
    # test_mask = graph.ndata["test_mask"]
    # labels = graph.ndata["label"]

    best_val_acc = 0
    best_val_epoch = 0
    best_model = None

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    for epoch in epoch_iter:
        model.train()
        out = model(None, x)
        ## ignore nan targets (unlabeled) when computing training loss.
        is_labeled = labels[train_mask] == labels[train_mask]
        #print(is_labeled)
        #print(out[train_mask][is_labeled])
        #print(torch.isnan(labels[train_mask][is_labeled]).any())
        loss = criterion(out[train_mask][is_labeled], labels[train_mask][is_labeled])
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
        optimizer.step()

        with torch.no_grad():
            model.eval()
            pred = model(None, x)
            # print(labels[val_mask].shape)
            # print(pred[val_mask].shape)
            # print(evaluator.eval({'y_true': labels[val_mask], 'y_pred': pred[val_mask]}))

            #print(pred[val_mask])
            val_acc = list(evaluator.eval({'y_true': labels[val_mask], 'y_pred': pred[val_mask]}).values())[0]
            #print(val_acc)
            #(pred[val_mask], labels[val_mask])
            is_labeled = labels[val_mask] == labels[val_mask]
            val_loss = criterion(pred[val_mask][is_labeled], labels[val_mask][is_labeled])
            test_acc = list(evaluator.eval({'y_true': labels[test_mask], 'y_pred': pred[test_mask]}).values())[0]
            #accuracy(pred[test_mask], labels[test_mask])
            is_labeled = labels[test_mask] == labels[test_mask]
            test_loss = criterion(pred[test_mask][is_labeled], labels[test_mask][is_labeled])
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_model = copy.deepcopy(model)

        if not mute:
            epoch_iter.set_description(f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, val_metric:{val_acc}, test_loss:{test_loss.item(): .4f}, test_metric:{test_acc: .4f}")

    best_model.eval()
    with torch.no_grad():
        pred = best_model(None, x)
        estp_test_acc = list(evaluator.eval({'y_true': labels[test_mask], 'y_pred': pred[test_mask]}).values())[0]
        #accuracy(pred[test_mask], labels[test_mask])
    if mute:
        print(f"# IGNORE: --- Testmetric: {test_acc:.4f}, early-stopping-Testmetric: {estp_test_acc:.4f}, Best Valmetric: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")
    else:
        print(f"--- Testmetric: {test_acc:.4f}, early-stopping-Testmetric: {estp_test_acc:.4f}, Best Valmetric: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")

    # (final_acc, es_acc, best_acc)
    return test_acc, estp_test_acc, best_val_acc

class LogisticRegression(nn.Module):
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)

    def forward(self, g, x, *args):
        logits = self.linear(x)
        return logits


class InContextDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset_name, args, emb, eval_tasks=None, num_label=None, num_support=None, num_query=None):
        super(InContextDataset, self).__init__()
        self.args = args
        self.device = args.device

        self.num_label = self.args.num_label if num_label is None else num_label
        self.num_support = self.args.num_support if num_support is None else num_support
        self.num_query = self.args.num_query if num_query is None else num_query

        if dataset_name in ['hiv']:
            self.eval_mol = Mol(self.args, dataset_name)
            link = None
            self.labels = self.eval_mol.labels
            self.split_idx = self.eval_mol.split_idx
        else:
            _, link, self.labels, self.split_idx = load_dataset(dataset_name)
        # feats, self.graph, self.labels, self.split_idx, ego_graph_nodes = load_large_dataset(dataset_name, args.data_dir)
        # self.graph.ndata["feat"] = feats
        if link is not None:
            node_pairs = torch.LongTensor(link["train"][0]+link["valid"][0]+link["test"][0])
            labels = torch.LongTensor(link["train"][1]+link["valid"][1]+link["test"][1])
            self.node_pairs = node_pairs
            self.labels = labels
            self.emb = torch.cat([emb[node_pairs[:, 0]], emb[node_pairs[:, 1]]], dim=1)
        else:
            self.emb = emb
        print(f"emb shape: {self.emb.shape}")


        print(f"Finish loading the data")
        # print(self.graph)
        self.label_dict = dict()
        for split in ["train", "valid", "test"]:
            for idx in self.split_idx[split]:
                label = self.labels[idx].item()
                if not np.isnan(label):
                    if label not in self.label_dict:
                        self.label_dict[label] = {"train": [], "valid": [], "test": []}
                    self.label_dict[label][split].append(idx)
        for label in self.label_dict.keys():
            print(label, len(self.label_dict[label]["train"]), len(self.label_dict[label]["valid"]), len(self.label_dict[label]["test"]))

        self.total_labels = len(self.label_dict.keys())
        #self.total_nodes = self.graph.num_nodes()
        if eval_tasks is None:
            self.total_steps = args.total_steps
        else:
            self.total_steps = eval_tasks
        
        # if self.args.drop_node > 0:
        #     self.drop_node = DropNode(p=self.args.drop_node)
    
    def generate_batch(self, batch_type="mt"):
        def sample(sample_list, size):
            if len(sample_list) >= size:
                return np.random.choice(sample_list, size=size, replace=False).tolist()
            return np.random.choice(sample_list, size=size, replace=True).tolist()

        m = self.num_label
        if batch_type == "mt":
            current_labels = np.random.choice(range(self.total_labels), m)
            while True:
                flag = True
                for label in current_labels:
                    if len(self.label_dict[label]["train"]) == 0 or len(self.label_dict[label]["test"]) == 0:
                        flag = False
                if flag:
                    break
                else:
                    current_labels = np.random.choice(range(self.total_labels), m)
        else:
            raise ValueError(batch_type)
        support_examples = []
        query_examples = []
        support_labels = []
        query_labels = []

        k = self.num_support
        n = self.num_query
        for idx, label in enumerate(current_labels):
            if batch_type == "mt":
                examples = sample(self.label_dict[label]["train"], k) + sample(self.label_dict[label]["test"], n)

            for i in range(k):
                support_examples.append(self.emb[examples[i]])
                support_labels.append(idx)

            for i in range(n):
                query_examples.append(self.emb[examples[k + i]])
                query_labels.append(idx)

        batch = {
            "support_examples": torch.stack(support_examples, dim=0),
            "query_examples": torch.stack(query_examples, dim=0),
            "support_labels": torch.LongTensor(support_labels),
            "query_labels": torch.LongTensor(query_labels),
        }

        return batch

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = max(worker_info.num_workers, 1)
        for _ in range(self.total_steps // num_workers):
            mt_batch = self.generate_batch(batch_type="mt")

            yield mt_batch

    def __len__(self):
        return self.total_steps

def setup_incontext_dataloader(emb, dataset_name, args):
    eval_dataset = InContextDataset(dataset_name, args, emb, eval_tasks=500, num_label=args.eval_num_label, num_support=args.eval_num_support, num_query=args.eval_num_query)
    dataloader = DataLoader(eval_dataset, batch_size=None, num_workers=4)

    return dataloader

def incontext_evaluate(args, emb, dataset_name):
    #device = args.device
    eval_dataloader = setup_incontext_dataloader(emb, dataset_name, args)
    acc_list = []
    for i, seed in enumerate(args.linear_prob_seeds):
        print(f"####### Run seed {seed} for In-Context Evaluation...")
        set_random_seed(seed)
        with torch.no_grad():
            index = []
            for i in range(args.eval_num_label):
                index.extend([i] * args.eval_num_support)
            index = torch.LongTensor(index).unsqueeze(1).expand(-1, emb.shape[1]*2 if dataset_name in ['FB15K237', 'WN18RR'] else emb.shape[1])
            accs = []
            for batch in tqdm(eval_dataloader):
                with torch.no_grad():

                    label_emb = torch.zeros(args.eval_num_label, emb.shape[1]*2 if dataset_name in ['FB15K237', 'WN18RR'] else emb.shape[1])
                    label_emb = label_emb.scatter_reduce(0, index, batch["support_examples"], reduce="mean")
                    query_emb = batch["query_examples"]

                    norm_query_emb = query_emb / torch.norm(query_emb, dim=1, keepdim=True)
                    norm_label_emb = label_emb / torch.norm(label_emb, dim=1, keepdim=True)

                    logits = torch.matmul(norm_query_emb, norm_label_emb.T)
                    acc = accuracy(logits, batch["query_labels"])
                    
                accs.append(acc)
            acc = np.mean(accs)
            print(f"# acc: {acc}")
        acc_list.append(acc)
    
    print(f"# acc: {np.mean(acc_list):.4f}±{np.std(acc_list):.4f}")
    acc = np.mean(acc_list)
    return acc

def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)

from sklearn.metrics import roc_auc_score

def calculate_aucroc(y_pred, y_true):
    y_prob = torch.sigmoid(y_pred[:, 1]).cpu().numpy()
    aucroc = roc_auc_score(y_true, y_prob)
    return aucroc