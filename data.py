from types import SimpleNamespace as SN
import os
import os.path as osp
from copy import deepcopy
from tqdm import tqdm
from torch.utils.data import DataLoader
import dgl
import numpy as np
import torch
import random
import itertools
from typing import Optional, Callable, List, Tuple, Any
from settings import *
from utils.preprocess import *
from utils.data_util import load_dataset
from tqdm import tqdm

from saint_sampler import get_saint_subgraphs
from dgl.data import DGLDataset
from torch_geometric.data import InMemoryDataset, Data
from transformers import AutoTokenizer

def safe_mkdir(path):
    if not osp.exists(path):
        os.mkdir(path)


def pth_safe_save(obj, path):
    if obj is not None:
        torch.save(obj, path)


def pth_safe_load(path):
    if osp.exists(path):
        return torch.load(path)
    return None

def get_ego_subgraphs(name):
    if name == "arxiv":
        ego_graphs = torch.load("./subgraphs/ogbn-arxiv-lc-ego-graphs-256.pt")
        return ego_graphs[0] + ego_graphs[1] + ego_graphs[2]
    elif name == "products":
        ego_graphs = torch.load("./subgraphs/ogbn-products-lc-ego-graphs-256.pt")
        return ego_graphs[0] + ego_graphs[1] + ego_graphs[2]
    elif name == "papers100M":
        ego_graphs = torch.load("./subgraphs/ogbn-papers100M-lc-ego-graphs-256.pt")
        return ego_graphs[0] + ego_graphs[1] + ego_graphs[2]
    
class TAG():
    def __init__(self, args, tag_name) -> None:
        self.name = tag_name
        self.data_info =  DATA_INFO[self.name]
        self.data = {}
        self.lm_type = args.lm_type
        self.token_folder = f"{DATA_PATH}{self.name}/{self.lm_type.split('/')[-1]}/"
        self.n_nodes = self.data_info['n_nodes']
        self.token_keys = ['attention_mask', 'input_ids', 'token_type_ids']
        self.mask_rate = args.mask_rate
        self.device = args.device
        self.args = args

        self.graph, self.test_graph, self.labels, self.split_idx = load_dataset(self.data_info["ogb_name"], args.task)
        # self.graph, self.labels, self.split_idx = load_ogb_graph_structure_only(args.dataset_name)
        # print(self.graph)

        self.info = {
            'input_ids': SN(shape=(self.data_info['n_nodes'], self.data_info['max_length']), type=np.uint16),
            'attention_mask': SN(shape=(self.data_info['n_nodes'], self.data_info['max_length']), type=bool),
            'token_type_ids': SN(shape=(self.data_info['n_nodes'], self.data_info['max_length']), type=bool)
        }
        for k, k_info in self.info.items():
            k_info.path = f'{self.token_folder}{k}.npy'

        self.load_data()

    def load_data_fields(self, k_list):
        for k in k_list:
            i = self.info[k]
            try:
                self.data[k] = np.memmap(i.path, mode='r', dtype=i.type, shape=i.shape)
                self.data[k] = self.data[k][:, :self.args.cut_off]
            except:
                raise ValueError(f'Shape not match {i.shape}')

    def get_tokens(self, node_id):
        _load = lambda k: torch.IntTensor(np.array(self.data[k][node_id]))
        item = {}
        item['attention_mask'] = _load('attention_mask')
        item['input_ids'] = torch.IntTensor(np.array(self.data['input_ids'][node_id]).astype(np.int32))
        if self.lm_type not in ['distilbert-base-uncased','roberta-base']:
            item['token_type_ids'] = _load('token_type_ids')
        return item

    def load_data(self):
        tokenize_graph(self.args, self.name)
        self.load_data_fields(self.token_keys)
        self.get_end_index()

    def get_end_index(self):
        if self.name == "papers100M":
            self.token_length_list = np.load('./dataset/ogbn_papers100M/token_length_list.npy').tolist()
        else:
            self.token_length_list = []
            for i in tqdm(range(self.n_nodes)):
                zero_index = torch.where(torch.IntTensor(np.array(self.data['input_ids'][i]).astype(np.int32))==0)[0]
                if len(zero_index) == 0:
                    end_index = len(self.data['input_ids'][i]) - 1
                else:
                    end_index = int(zero_index[0] - 1)
                token_length = end_index # - 1
                self.token_length_list.append(token_length)
            token_length_list=np.array(self.token_length_list)

class IterTAGDataset(torch.utils.data.IterableDataset):  # Iterable style
    def __init__(self, data: TAG, idx, batch_size=None, num_roots=100, length=4):
        self.data = data
        self.idx = idx
        # self.batch_size = batch_size
    
        # get mask token id
        tokenizer = AutoTokenizer.from_pretrained(self.data.lm_type)
        self.mask_token_id = tokenizer(tokenizer.mask_token)['input_ids'][1]

        #self.loader = get_saint_subgraphs(self.data.graph, self.data.data_info["ogb_name"], num_roots=num_roots, length=length)
        self.loader = get_ego_subgraphs(self.data.name)
        #print(len(self.loader))
        #print(self.loader[0])
        # print(self.loader)
        self.batch_size = batch_size

    def __getitem__(self, node_id):
        item = self.data.get_tokens(node_id)
        masked_input_ids = self.get_masked_item(item, self.data.token_length_list[node_id])
        return item, masked_input_ids, self.idx

    def get_masked_item(self, item, token_length):
        if token_length <= 0:
            masked_input_ids = item['input_ids'].clone()
        else:
            mask_num = int(token_length * self.data.mask_rate)
            # print(mask_num, token_length)
            mask_list = random.sample(range(1, token_length + 1), mask_num)
        # except:
        #     mask_list = [1]
        #     print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> token_length {token_length}")
        #     print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> mask_num {mask_num}")
            masked_input_ids = item['input_ids'].index_fill(0, torch.tensor(mask_list, dtype=torch.long), self.mask_token_id)
        return masked_input_ids

    def __iter__(self):
        for iter_data in self.loader:
            #print(len(iter_data))
            #print(iter_data)
            if self.batch_size is not None:
                iter_data = iter_data[:self.batch_size]
            batch = {"input_ids": [], "attention_mask": [], "token_type_ids": []}
            batch["masked_input_ids"] = []
            #batch["idx"] = []
            # masked_batch_ids = []
            for key in iter_data:
                item, masked_input_ids, idx = self.__getitem__(key)
                batch["input_ids"].append(item["input_ids"])
                batch["attention_mask"].append(item["attention_mask"])
                batch["token_type_ids"].append(item["token_type_ids"])
                batch["masked_input_ids"].append(masked_input_ids)
                #batch["idx"].append(idx)
                # masked_batch_ids.append(masked_input_ids)

            batch["input_ids"] = torch.stack(batch["input_ids"], 0)
            batch["attention_mask"] = torch.stack(batch["attention_mask"], 0)
            batch["token_type_ids"] = torch.stack(batch["token_type_ids"], 0)
            batch["masked_input_ids"] = torch.stack(batch["masked_input_ids"], 0)
            #batch["idx"] = torch.tensor(batch["idx"], dtype=torch.long)
            # print(iter_data, batch)
            # masked_batch_ids = torch.cat(masked_batch_ids, 0)
            # yield batch, masked_batch_ids
            
            # graph = dgl.node_subgraph(self.data.graph, iter_data)
            yield batch, iter_data, self.idx

    def __len__(self):
        return len(self.loader)
    
class Mol(InMemoryDataset):
    def __init__(self, args, name: str, root: str = "./dataset", load_text: bool = True,
                 transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None):
        self.name = name
        self.root = root
        self.args = args
        self.lm_type = args.lm_type
        self.mask_rate = args.mask_rate
        self.data_dir = osp.join(self.root, self.name)
        super().__init__(self.data_dir, transform, pre_transform)
        #safe_mkdir(self.data_dir)


        self.tokenizer = AutoTokenizer.from_pretrained(self.lm_type)

        # load and tokenize text to the dataset instance
        if load_text:
            self.texts = self.tokenize_texts(torch.load(self.processed_paths[1]))[0]
            self.texts = {k: np.array(v)[:, :self.args.cut_off//2] for k, v in self.texts.items()}

        self.num_tasks = 1 if self.name == "hiv" else 128
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.split_idx = torch.load(self.processed_paths[2])[0]
        self.labels = torch.load(self.processed_paths[0])[0].y.reshape(-1, self.num_tasks)
        #self.side_data = pth_safe_load(self.processed_paths[2])
        self.calculate_token_lengths()

    @property
    def num_classes(self):
        return self.__num_classes__

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["geometric_data_processed.pt", "texts.pkl", "data.pt"]

    def tokenize_texts(self, texts):
        return [self.tokenizer(text, padding='max_length', truncation=True, max_length=512, return_token_type_ids=True).data
                for text in texts]
    
    def calculate_token_lengths(self):
        self.token_length_list = []
        for text in self.texts['input_ids']:
            #zero_index = torch.where(text == 0)[0]
            zero_index = torch.where(torch.IntTensor(np.array(text).astype(np.int32))==0)[0]
            if len(zero_index) == 0:
                end_index = len(text) - 1
            else:
                end_index = int(zero_index[0] - 1)
            token_length = end_index
            self.token_length_list.append(token_length)

    def get_tokens(self, node_id):
        _load = lambda k: torch.IntTensor(np.array(self.texts[k][node_id]))
        item = {}
        item['attention_mask'] = _load('attention_mask')
        item['input_ids'] = torch.IntTensor(np.array(self.texts['input_ids'][node_id]).astype(np.int32))
        if self.lm_type not in ['distilbert-base-uncased','roberta-base']:
            item['token_type_ids'] = _load('token_type_ids')
        return item

    def get(self, index):
        data = super().get(index)
        #node_feat = self.node_embs[data.x]
        #data.edge_feat = self.edge_embs[data.xe]
        # Replace text_feat with tokenized text
        #print(data.x)
        data.text_feat = self.get_tokens(data.x)
        #{k: v[data.x] for k, v in self.texts.items()}
        #self.texts[data.x]
        #data.node_text_feat = node_feat
        #data.edge_text_feat = edge_feat
        data.y = data.y.view(-1)
        return data
    
class IterMolDataset(torch.utils.data.IterableDataset):  # Iterable style
    def __init__(self, data: Mol, idx, batch_size=None):
        self.data = data
        self.idx = idx
        # self.batch_size = batch_size
    
        # get mask token id
        tokenizer = AutoTokenizer.from_pretrained(self.data.lm_type)
        self.mask_token_id = tokenizer(tokenizer.mask_token)['input_ids'][1]

        #self.loader = get_saint_subgraphs(self.data.graph, self.data.data_info["ogb_name"], num_roots=num_roots, length=length)
        #self.loader = get_ego_subgraphs(self.data.name)
        print(self.data)
        self.loader = self.data
        #self.loader = range(len(self.data))
        #print(len(self.loader))
        #print(self.loader[0])
        print(self.loader)
        #self.batch_size = batch_size
    
    def __getitem__(self, graph):
        # Process each node in graph.x
        item = graph.text_feat
        masked_items = []
        for idx, node_id in enumerate(graph.x):
            unmasked_input_ids = graph.text_feat['input_ids'][idx]
            token_length = self.data.token_length_list[node_id.item()]
            masked_input_ids = self.get_masked_item(unmasked_input_ids, token_length)
            masked_items.append(masked_input_ids)

        return item, masked_items, self.idx

    def get_masked_item(self, item, token_length):
        if token_length <= 0:
            masked_input_ids = item.clone()
        else:
            mask_num = int(token_length * self.data.mask_rate)
            # print(mask_num, token_length)
            mask_list = random.sample(range(1, token_length + 1), mask_num)
        # except:
        #     mask_list = [1]
        #     print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> token_length {token_length}")
        #     print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> mask_num {mask_num}")
            masked_input_ids = item.index_fill(0, torch.tensor(mask_list, dtype=torch.long), self.mask_token_id)
        return masked_input_ids

    def __iter__(self):
        for iter_data in self.loader:
            #print(len(iter_data))
            #print(iter_data)
            #if self.batch_size is not None:
            #    iter_data = iter_data[:self.batch_size]
            batch = {"input_ids": [], "attention_mask": [], "token_type_ids": []}
            batch["masked_input_ids"] = []
            #batch["idx"] = []
            # masked_batch_ids = []
            #for key in iter_data:
            item, masked_input_ids, idx = self.__getitem__(iter_data)
            batch["input_ids"] = item["input_ids"].reshape(-1, self.data.args.cut_off//2) 
            batch["attention_mask"] = item["attention_mask"].reshape(-1, self.data.args.cut_off//2) 
            batch["token_type_ids"] = item["token_type_ids"].reshape(-1, self.data.args.cut_off//2) 
            batch["masked_input_ids"] = torch.stack(masked_input_ids, 0)
            graph = dgl.DGLGraph()
            graph.add_nodes(iter_data.x.shape[0])
            graph.add_edges(iter_data.edge_index[0], iter_data.edge_index[1])
            #graph.edata['feat'] = iter_data.edge_feat
            graph = graph.remove_self_loop().add_self_loop()
            graph.y = iter_data.y
            iter_data = graph
                #batch["idx"].append(idx)
                # masked_batch_ids.append(masked_input_ids)

            # batch["input_ids"] = torch.stack(batch["input_ids"], 0)
            # batch["attention_mask"] = torch.stack(batch["attention_mask"], 0)
            # batch["token_type_ids"] = torch.stack(batch["token_type_ids"], 0)
            # batch["masked_input_ids"] = torch.stack(batch["masked_input_ids"], 0)
            #batch["idx"] = torch.tensor(batch["idx"], dtype=torch.long)
            # print(iter_data, batch)
            # masked_batch_ids = torch.cat(masked_batch_ids, 0)
            # yield batch, masked_batch_ids
            
            # graph = dgl.node_subgraph(self.data.graph, iter_data)
            yield batch, iter_data, self.idx

    def __len__(self):
        return len(self.loader)
    
class CombinedDataset(torch.utils.data.IterableDataset):
    def __init__(self, datasets, batch_size=None):
        """
        Initializes a CombinedDataset object.

        :param datasets: A list of IterTAGDataset instances to be combined.
        :param batch_size: The batch size for each iteration, if applicable.
        """
        #assert all(isinstance(dataset, IterTAGDataset) for dataset in datasets), "All elements must be IterTAGDataset instances"
        self.datasets = datasets
        self.batch_size = batch_size
        if datasets:
            self.lm_type = datasets[0].data.lm_type
            self.mask_token_id = self._get_mask_token_id(self.lm_type)

    def _get_mask_token_id(self, lm_type):
        """
        获取 mask token ID.

        :param lm_type: 用于 tokenizer 的语言模型类型。
        :return: mask token ID.
        """
        tokenizer = AutoTokenizer.from_pretrained(lm_type)
        return tokenizer(tokenizer.mask_token)['input_ids'][1]

    # def __iter__(self):
    #     """
    #     Provides an iterator over all the datasets.

    #     This method combines the data from all the IterTAGDataset instances,
    #     yielding batches of data in a round-robin fashion from each dataset.
    #     """
    #     iterators = [iter(dataset) for dataset in self.datasets]
    #     while iterators:
    #         for iterator in list(iterators):  # Iterate over a copy of the list to allow safe removal
    #             try:
    #                 batch, iter_data, idx = next(iterator)
    #                 yield batch, iter_data, idx
    #             except StopIteration:
    #                 iterators.remove(iterator)

    def __iter__(self):
        """
        Provides an iterator over all the datasets.

        This method combines the data from all the IterTAGDataset instances,
        randomly selecting and yielding batches of data from one of the datasets at each step.
        """
        iterators = [iter(dataset) for dataset in self.datasets]
        while iterators:
            # Randomly select an iterator
            iterator = random.choice(iterators)
            try:
                batch, iter_data, idx = next(iterator)
                yield batch, iter_data, idx
            except StopIteration:
                # Remove the exhausted iterator
                iterators.remove(iterator)
                if not iterators:
                    break
                # Continue with a new random choice from the remaining iterators
                iterator = random.choice(iterators)

    # def __iter__(self):
    #     """
    #     Provides an iterator over all the datasets.

    #     This method iterates over each dataset, draws one item from each,
    #     shuffles these items, and then yields them in random order.
    #     """
    #     iterators = [iter(dataset) for dataset in self.datasets]
        
    #     while iterators:
    #         batch = []
    #         # Remove exhausted iterators
    #         iterators = [it for it in iterators if not self.is_iterator_exhausted(it)]

    #         if not iterators:
    #             break

    #         for iterator in iterators:
    #             try:
    #                 item = next(iterator)
    #                 batch.append(item)
    #             except StopIteration:
    #                 # Handle case where an iterator gets exhausted
    #                 pass

    #         random.shuffle(batch)
    #         for item in batch:
    #             yield item

    # def is_iterator_exhausted(self, iterator):
    #     """
    #     Checks if an iterator is exhausted.
    #     """
    #     try:
    #         # Peek next item
    #         next(iterator)
    #         # Restore the item back to the iterator
    #         iterator = itertools.chain([next(iterator)], iterator)
    #         return False
    #     except StopIteration:
    #         return True

    def __len__(self):
        """
        Returns the total number of iterations possible with the combined datasets.

        This is the sum of the lengths of all individual datasets.
        """
        return sum(len(dataset) for dataset in self.datasets)

class TAGDataset(torch.utils.data.Dataset):  # Map style
    def __init__(self, data: TAG):
        self.data = data
        # get mask token id
        tokenizer = AutoTokenizer.from_pretrained(self.data.lm_type)
        self.mask_token_id = tokenizer(tokenizer.mask_token)['input_ids'][1]

    def __getitem__(self, node_id):
        item = self.data.get_tokens(node_id)
        masked_input_ids = self.get_masked_item(item, self.data.token_length_list[node_id])
        return item, masked_input_ids

    def get_masked_item(self, item, token_length):
        if token_length <= 0:
            masked_input_ids = item['input_ids'].clone()
        else:
            mask_num = int(token_length * self.data.mask_rate)
            # print(mask_num, token_length)
            mask_list = random.sample(range(1, token_length + 1), mask_num)
        # except:
        #     mask_list = [1]
        #     print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> token_length {token_length}")
        #     print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> mask_num {mask_num}")
            masked_input_ids = item['input_ids'].index_fill(0, torch.tensor(mask_list, dtype=torch.long), self.mask_token_id)
        return masked_input_ids
        # masked_item = item.copy()
        # # masked_item['input_ids'] = deepcopy(item['input_ids'])
        # # token_length = item['token_length']
        # mask_num = int(token_length * self.data.mask_rate)
        # mask_list = random.sample(range(1, token_length + 1), mask_num)
        # masked_item['input_ids'] = masked_item['input_ids'].index_fill(0, torch.tensor(mask_list), self.mask_token_id)
        # return masked_item
    # def get_masked_batch(self, batch):
    #     masked_batch = batch
    #     masked_batch['input_ids'] = deepcopy(batch['input_ids'])
    #     for i in range(len(masked_batch['input_ids'])):
    #         token_length = masked_batch['token_length'][i]
    #         mask_num = int(token_length * self.mask_rate)
    #         mask_list = random.sample(range(1, token_length + 1), mask_num)
    #         masked_batch['input_ids'][i] = masked_batch['input_ids'][i].index_fill(0, torch.tensor(mask_list).to(self.device), self.mask_token_id)
    #     return masked_batch

    def __len__(self):
        return self.data.n_nodes
