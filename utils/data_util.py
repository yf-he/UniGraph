from collections import namedtuple, Counter
import numpy as np

import torch
import torch.nn.functional as F

import dgl
from dgl.data import (
    load_data, 
    TUDataset, 
    CoraGraphDataset, 
    CiteseerGraphDataset, 
    PubmedGraphDataset
)
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data.ppi import PPIDataset
from dgl.dataloading import GraphDataLoader
from dgl.data import load_graphs, save_graphs

from sklearn.preprocessing import StandardScaler
from data_tg import TextualGraphDataset

GRAPH_DICT = {
    "cora": CoraGraphDataset,
    "citeseer": CiteseerGraphDataset,
    "pubmed": PubmedGraphDataset,
    "ogbn-arxiv": DglNodePropPredDataset,
    "ogbn-products": DglNodePropPredDataset,
    "ogbn-papers100M": DglNodePropPredDataset,
    "cora_ml":TextualGraphDataset,
}


def preprocess(graph):
    # feat = graph.ndata["feat"]
    if "feat" in graph.ndata:
        graph.ndata.pop("feat") 
    graph = dgl.to_bidirected(graph)
    # graph.ndata["feat"] = feat

    graph = graph.remove_self_loop().add_self_loop()
    #graph.create_formats_()
    return graph


def scale_feats(x):
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats


def load_dataset(dataset_name, task="nc"):
    #assert dataset_name in GRAPH_DICT, f"Unknow dataset: {dataset_name}."
    if dataset_name == 'arxiv':
        dataset_name = 'ogbn-arxiv'
    if dataset_name.startswith("ogbn"):
        if dataset_name == 'ogbn-papers100M':
            # dataset = GRAPH_DICT[dataset_name](dataset_name, root=f"./dataset")
            # print("Saving graph to ./papergraph.bin")
            # graph, labels = dataset[0]
            # print(graph)
            # feats = graph.ndata.pop("feat")
            # year = graph.ndata.pop("year")
            # #graph = preprocess(graph)
            # print(graph)
            # save_graphs("./papergraph.bin", graph, {"labels": labels.view(-1)})
            print("Loading graph from ./papergraph.bin")
            graph, labels = load_graphs("./dataset/ogbn_papers100M/papergraph.bin")
            print("Done")
            graph = graph[0]
            print(graph)
            test_graph = None
            split_idx = None
        else:
            dataset = GRAPH_DICT[dataset_name](dataset_name, root=f"./dataset")
    elif dataset_name in ["cora_ml"]:
        dataset = GRAPH_DICT[dataset_name](dataset_name, root=f"./dataset/", task=task)
    elif dataset_name in ["wikics"]:
        return load_wikics_dataset(dataset_name)
    elif dataset_name in ["FB15K237", "WN18RR"]:
        return load_kg_dataset(dataset_name)
    elif dataset_name in ["cora", "pubmed"]:
        return load_citation_dataset(dataset_name)
    else:
        dataset = GRAPH_DICT[dataset_name]()

    # if dataset_name == "ogbn-arxiv":
    #     graph, labels = dataset[0]
    #     num_nodes = graph.num_nodes()

    #     split_idx = dataset.get_idx_split()
    #     train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    #     graph = preprocess(graph)

    #     if not torch.is_tensor(train_idx):
    #         train_idx = torch.as_tensor(train_idx)
    #         val_idx = torch.as_tensor(val_idx)
    #         test_idx = torch.as_tensor(test_idx)

    #     feat = graph.ndata["feat"]
    #     feat = scale_feats(feat)
    #     graph.ndata["feat"] = feat

    #     train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx, True)
    #     val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx, True)
    #     test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True)
    #     graph.ndata["label"] = labels.view(-1)
    #     graph.ndata["train_mask"], graph.ndata["val_mask"], graph.ndata["test_mask"] = train_mask, val_mask, test_mask
    if dataset_name in ["cora_ml", 'ogbn-arxiv', 'ogbn-products']:
        graph, labels = dataset[0]

        if "year" in graph.ndata:
            del graph.ndata["year"]
        if not graph.is_multigraph:
            graph = preprocess(graph)
        else:
            graph = graph.remove_self_loop().add_self_loop()

        split_idx = dataset.get_idx_split()
        if labels is not None:
            labels = labels.view(-1)
    if task == "lp":
        test_graph = dataset.test_graph
    else:
        test_graph = None
        # feats = graph.ndata.pop("feat") 
        # if dataset_name in ("ogbn-arxiv","ogbn-papers100M"):
        #     feats = scale_feats(feats)
    # else:
    #     graph = dataset[0]
    #     graph = graph.remove_self_loop()
    #     graph = graph.add_self_loop()
    # num_features = graph.ndata["feat"].shape[1]
    # num_classes = dataset.num_classes
    # return graph, (num_features, num_classes)
    return graph, test_graph, labels, split_idx

def load_wikics_dataset(dataset_name):
    data = torch.load("./dataset/wikics/processed/data_undirected.pt")[0]
    graph = dgl.DGLGraph()
    graph.add_nodes(11701)
    graph.add_edges(data.edge_index[0], data.edge_index[1])
    graph = preprocess(graph)
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")
    print(graph)
    labels = data.y
    test_graph = None
    train_idx = data.train_mask[:,0].nonzero().squeeze(1)
    val_idx = data.val_mask[:,0].nonzero().squeeze(1)
    test_idx = data.test_mask.nonzero().squeeze(1)
    split_idx = {"train": train_idx, "valid": val_idx, "test": test_idx}
    print(f"Train: {len(train_idx)}, Valid: {len(val_idx)}, Test: {len(test_idx)}")
    return graph, test_graph, labels, split_idx

def load_citation_dataset(dataset_name):
    data = torch.load(f"./dataset/{dataset_name}/processed/geometric_data_processed.pt")[0]
    graph = dgl.DGLGraph()
    graph.add_nodes(data.x.shape[0])
    graph.add_edges(data.edge_index[0], data.edge_index[1])
    graph = preprocess(graph)
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")
    print(graph)
    labels = data.y
    test_graph = None
    train_idx = data.train_masks[0].nonzero().squeeze(1)
    val_idx = data.val_masks[0].nonzero().squeeze(1)
    test_idx = data.test_masks[0].nonzero().squeeze(1)
    split_idx = {"train": train_idx, "valid": val_idx, "test": test_idx}
    print(f"Train: {len(train_idx)}, Valid: {len(val_idx)}, Test: {len(test_idx)}")
    return graph, test_graph, labels, split_idx

def load_kg_dataset(dataset_name):
    data = torch.load(f"./dataset/{dataset_name}/processed/geometric_data_processed.pt")[0]
    graph = dgl.DGLGraph()
    graph.add_nodes(data.x.shape[0])
    graph.add_edges(data.edge_index[0], data.edge_index[1])
    #graph.edata["feat"] = data.edge_text_feat[data.edge_types]
    graph = preprocess(graph)
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")
    print(graph)
    #print(graph.edata["feat"][-1])
    labels = data.y
    converted_triplet = torch.load(f"./dataset/{dataset_name}/processed/data.pt")[0]
    split_idx = {}
    count = 0
    for name in converted_triplet:
        split_idx[name] = torch.arange(count, count + len(converted_triplet[name][0]))
        count += len(converted_triplet[name][0])
    test_graph = converted_triplet
    print(f"Train: {len(split_idx['train'])}, Valid: {len(split_idx['valid'])}, Test: {len(split_idx['test'])}")
    print(split_idx['train'])
    print(split_idx['valid'])
    print(split_idx['test'])
    print(f"Train: {len(test_graph['train'][0])}, Valid: {len(test_graph['valid'][0])}, Test: {len(test_graph['test'][0])}")
    print(f"Train: {len(test_graph['train'][1])}, Valid: {len(test_graph['valid'][1])}, Test: {len(test_graph['test'][1])}")
    return graph, test_graph, labels, split_idx

    dataset_name = dataset_name.upper()
    dataset = TUDataset(dataset_name)
    graph, _ = dataset[0]

    if "attr" not in graph.ndata:
        if "node_labels" in graph.ndata and not deg4feat:
            print("Use node label as node features")
            feature_dim = 0
            for g, _ in dataset:
                feature_dim = max(feature_dim, g.ndata["node_labels"].max().item())
            
            feature_dim += 1
            for g, l in dataset:
                node_label = g.ndata["node_labels"].view(-1)
                feat = F.one_hot(node_label, num_classes=feature_dim).float()
                g.ndata["attr"] = feat
        else:
            print("Using degree as node features")
            feature_dim = 0
            degrees = []
            for g, _ in dataset:
                feature_dim = max(feature_dim, g.in_degrees().max().item())
                degrees.extend(g.in_degrees().tolist())
            MAX_DEGREES = 400

            oversize = 0
            for d, n in Counter(degrees).items():
                if d > MAX_DEGREES:
                    oversize += n
            # print(f"N > {MAX_DEGREES}, #NUM: {oversize}, ratio: {oversize/sum(degrees):.8f}")
            feature_dim = min(feature_dim, MAX_DEGREES)

            feature_dim += 1
            for g, l in dataset:
                degrees = g.in_degrees()
                degrees[degrees > MAX_DEGREES] = MAX_DEGREES
                
                feat = F.one_hot(degrees, num_classes=feature_dim).float()
                g.ndata["attr"] = feat
    else:
        print("******** Use `attr` as node features ********")
        feature_dim = graph.ndata["attr"].shape[1]

    labels = torch.tensor([x[1] for x in dataset])
    
    num_classes = torch.max(labels).item() + 1
    dataset = [(g.remove_self_loop().add_self_loop(), y) for g, y in dataset]

    print(f"******** # Num Graphs: {len(dataset)}, # Num Feat: {feature_dim}, # Num Classes: {num_classes} ********")

    return dataset, (feature_dim, num_classes)