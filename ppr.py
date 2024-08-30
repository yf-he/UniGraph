import argparse
import numba
import numpy as np
import torch
import logging
import os
import scipy.sparse as sp

import dgl
from dgl.data import load_data, CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from ogb.nodeproppred import DglNodePropPredDataset

def collect_topk_ppr(graph, nodes, topk, alpha, epsilon):
    if torch.is_tensor(nodes):
        nodes = nodes.numpy()
    row, col = graph.edges()
    row = row.numpy()
    col = col.numpy()
    num_nodes = graph.num_nodes()

    neighbors = build_topk_ppr((row, col), alpha, epsilon, nodes, topk, num_nodes=num_nodes)
    return neighbors

# modified
@numba.njit(cache=True, locals={"_val": numba.float32, "res": numba.float32, "res_vnode": numba.float32})
def _calc_ppr_node(inode, indptr, indices, deg, alpha, epsilon):
    alpha_eps = alpha * epsilon
    f32_0 = numba.float32(0)
    p = {inode: f32_0}
    r = {inode: numba.float32(1.0)}
    q = [inode]
    while len(q) > 0:
        unode = q.pop()

        res = r[unode] if unode in r else f32_0
        if unode in p:
            p[unode] += alpha * res
        else:
            p[unode] = alpha * res
        r[unode] = f32_0
        for vnode in indices[indptr[unode]: indptr[unode + 1]]:
            _val = (1 - alpha) * res / deg[unode]
            if vnode in r:
                r[vnode] += _val
            else:
                r[vnode] = _val

            res_vnode = r[vnode] if vnode in r else f32_0
            if res_vnode >= alpha_eps * deg[vnode]:
                if vnode not in q:
                    q.append(vnode)

    return list(p.keys()), list(p.values())





@numba.njit(cache=True, parallel=True)
def calc_ppr_topk_parallel(indptr, indices, deg, alpha, epsilon, nodes, topk, mode="transformer"):
    js = [np.zeros(0, dtype=np.int64)] * len(nodes)
    vals = [np.zeros(0, dtype=np.float32)] * len(nodes)
    self_val = np.array([1.0]).astype(vals[0].dtype)
    mask_val = np.array([-1], dtype=js[0].dtype)

    for i in numba.prange(len(nodes)):
        j, val = _calc_ppr_node(nodes[i], indptr, indices, deg, alpha, epsilon)
        j_np, val_np = np.array(j), np.array(val)
        idx_topk = np.argsort(val_np)[-topk:][::-1]

        js[i] = j_np[idx_topk]
        vals[i] = val_np[idx_topk]

        mask = (js[i] != nodes[i])
        js[i] = js[i][mask]
        vals[i] = vals[i][mask]

        # if mode == "transformer" and len(js[i]) < topk:
        #     diff = topk - len(js[i])
        #     js[i] = np.concatenate((np.array([nodes[i]], dtype=js[0].dtype), js[i], mask_val.repeat(diff)))
        #     vals[i] = np.concatenate((self_val, vals[i], self_val.repeat(diff)))
        # else:
        js[i] = np.concatenate((np.array([nodes[i]]), js[i]))
        vals[i] = np.concatenate((self_val, vals[i]))
    return js, vals


@numba.njit(cache=True, parallel=True)
def add_padding(js, vals, topk, nodes):
    v_type = vals[0].dtype
    self_val = np.array([1.0], dtype=v_type)
    mask_val = np.array([-1], dtype=js[0].dtype)
    for i in numba.prange(len(nodes)):
        if len(js[i]) < topk:
            diff = topk - len(js[i])
            js[i] = np.concatenate((np.array([i], dtype=js[0].dtype), js[i], mask_val.repeat(diff)))
            vals[i] = np.concatenate((self_val, vals[i], self_val.reshape(diff)))
    return js, vals


# def pos_embedding(neighbors, adj, hidden_size):
#     nodes = neighbors
#     mx = adj[nodes, :][:, nodes]
#     emb = get_undirected_graph_positional_embedding(mx, hidden_size)
#     return emb


def ppr_ego_graphs(adj_matrix, alpha, epsilon, nodes, topk, mode="transformer"):
    out_degree = np.sum(adj_matrix > 0, axis=1).A1
    neighbors, weights = calc_ppr_topk_parallel(
        adj_matrix.indptr, adj_matrix.indices, out_degree, numba.float32(alpha), numba.float32(epsilon), nodes, topk, mode
    )
    for i in range(len(weights[0]) - 1):
        assert weights[0][i] >= weights[0][i+1]
    return neighbors, weights


def build_topk_ppr(edge_index, alpha, epsilon, nodes, topk, mode="transformer", num_nodes=-1):
    assert num_nodes > 0

    val = np.ones(edge_index[0].shape[0])
    adj_matrix = sp.csr_matrix((val, (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes))
    neighbors, weights = ppr_ego_graphs(adj_matrix, alpha, epsilon, nodes, topk, mode)
    return neighbors

def lc_prepare_ego_graphs(split_idx,dataset_name, graph, topk, alpha, epsilon, nodes=None, save_dir="./lc_preprocessed"):
    save_dir = "./data/ego-graphs"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, dataset_name), exist_ok=True)
    if nodes is not None:
        suffix = "ppr_eval"
    else:
        suffix = "ppr"
    save_path = os.path.join(save_dir, dataset_name, f"k-{topk}_alpha-{alpha}_eps_{epsilon}_{suffix}.pt")

    if nodes is None:
        #split_idx = dataset.get_idx_split()
        if dataset_name not in ["cora", "citeseer", "pubmed","film"]:
            train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
            num_used_nodes = train_idx.shape[0] + val_idx.shape[0] + test_idx.shape[0]
        else:
            num_used_nodes = graph.num_nodes()

        if num_used_nodes < graph.num_nodes():
            logging.info("use 10 times unlabelled nodes ..")
            labelled_nodes = torch.cat([train_idx, val_idx, test_idx], dim=0)
            sample_nodes = np.random.randint(0, int(graph.num_nodes()), 10*num_used_nodes)
            sample_nodes = np.setdiff1d(sample_nodes, labelled_nodes)
            sample_nodes = torch.from_numpy(sample_nodes)
            nodes = torch.cat([labelled_nodes, sample_nodes], dim=0)
        else:
            logging.info("use full nodes")
            nodes = torch.arange(graph.num_nodes()).numpy()
    else:
        if not torch.is_tensor(nodes):
            nodes = torch.tensor(nodes)

    if os.path.exists(save_path):
        logging.info("--- start loading graphs ---")
        ego_graph_nodes = torch.load(save_path)
    else:
        logging.info("--- computing ppr ---")
        ego_graph_nodes = collect_topk_ppr(graph, nodes, topk, alpha, epsilon)
        logging.info("--- finish computing ppr ---")
        torch.save(ego_graph_nodes, save_path)
    logging.info("--- ego-graph loaded ---")

    avg_size = np.mean([x.shape[0] for x in ego_graph_nodes])
    logging.info(f"Average ego-graph size: {avg_size:.2f}")

    logging.info("--- preprocessing done ---")
    return ego_graph_nodes

def preprocess(graph):
    # make bidirected
    if "feat" in graph.ndata:
        feat = graph.ndata["feat"]
    else:
        feat = None
    graph = dgl.to_bidirected(graph)
    if feat is not None:
        graph.ndata["feat"] = feat

    # add self-loop
    graph = graph.remove_self_loop().add_self_loop()
    # graph.create_formats_()
    return graph

def load_dataset(dataset_name):
    start_path = ""
    if dataset_name.startswith("ogbn"):
        dataset = DglNodePropPredDataset(dataset_name,root=os.path.join(start_path, "dataset"))
        graph, label = dataset[0]

        if "year" in graph.ndata:
            del graph.ndata["year"]
        if not graph.is_multigraph:
            graph = preprocess(graph)
        # graph = graph.remove_self_loop().add_self_loop()

        split_idx = dataset.get_idx_split()
        label = label.view(-1)

        feats = graph.ndata.pop("feat")


    return feats, graph, label, split_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--task', type=str, default='saint')
    parser.add_argument('--dataset', type=str, default='flickr')
    parser.add_argument('--ego_size', type=int, default=1024)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_iter', type=int, default=1000)
    parser.add_argument('--log_steps', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--method', type=str, default='acl')
    parser.add_argument('--num_workers', type=int, default=50)
    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)

    feat, graph, label, split_idx = load_dataset(args.dataset)
    ego_graph_nodes = lc_prepare_ego_graphs(split_idx, args.dataset, graph, args.ego_size, alpha=0.75, epsilon=0.000001)