import os
import torch
import logging
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModel, get_scheduler
from tqdm.auto import tqdm
import wandb
import dgl

from data import *
from model import UniGraph
from utils.functions import create_optimizer, get_current_lr, set_random_seed, drop_edge, pool, Evaluator
from utils.evaluation import node_classification_evaluation, link_prediction_evaluation, edge_classification_evaluation, graph_classification_evaluation, incontext_evaluate
from utils.data_util import preprocess
from gensim.models import KeyedVectors

def evaluate(args, model, device, name=""):
    for dataset in args.eval_datasets_name:
        if dataset in ['cora', 'pubmed', 'arxiv', 'products', 'wikics', 'FB15K237', 'WN18RR']:
            evaluate_tag(args, model, device, dataset)
        elif args.task in ["w2cnc", "w2cec"]:
            evaluate_w2v(args, model, device, dataset)
        else:
            evaluate_mol(args, model, device, dataset)
    return

def evaluate_tag(args, model, device, name=""):
    eval_tag = TAG(args, name)
    dataset = TAGDataset(eval_tag)
    eval_loader = DataLoader(dataset, shuffle=False, batch_size=args.eval_batch_size)
    model.eval()
    output = []

    for batch, _ in tqdm(eval_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        emb = model.emb(batch)
        output.append(emb.cpu())
    output = torch.cat(output, 0)

    if args.incontext_eval:
        acc = incontext_evaluate(args, output, name)
        graph = eval_tag.graph
        graph.ndata["feat"] = output
        output, cat_output = model.inference(graph, device, args.eval_batch_size)
        acc = incontext_evaluate(args, output, name)
        acc = incontext_evaluate(args, cat_output, name)

    if name in ['cora', 'pubmed', 'arxiv', 'products', 'wikics']:
        graph = eval_tag.graph
        test_acc, estp_test_acc, best_val_acc = node_classification_evaluation(graph, output, eval_tag.labels, eval_tag.split_idx, eval_tag.data_info["n_labels"], args.lr_f, args.weight_decay_f, args.lp_epochs, device)
        wandb.log({
            f"{name}_estp_test_acc_lm": estp_test_acc,
            f"{name}_best_val_acc_lm": best_val_acc,
        })

        if args.gnn_type != "":
            graph.ndata["feat"] = output
            output, cat_output = model.inference(graph, device, args.eval_batch_size)
            test_acc, estp_test_acc, best_val_acc = node_classification_evaluation(graph, output, eval_tag.labels, eval_tag.split_idx, eval_tag.data_info["n_labels"], args.lr_f, args.weight_decay_f, args.lp_epochs, device)
            wandb.log({
                f"{name}_estp_test_acc_gnn": estp_test_acc,
                f"{name}_best_val_acc_gnn": best_val_acc,
            })

    elif name in ['FB15K237', 'WN18RR']:
        graph = eval_tag.graph
        node_pairs = torch.LongTensor(eval_tag.test_graph["train"][0] + eval_tag.test_graph["valid"][0] + eval_tag.test_graph["test"][0])
        labels = torch.LongTensor(eval_tag.test_graph["train"][1] + eval_tag.test_graph["valid"][1] + eval_tag.test_graph["test"][1])
        test_acc, estp_test_acc, best_val_acc = edge_classification_evaluation(graph, output, node_pairs, labels, eval_tag.split_idx, eval_tag.data_info["n_labels"], args.lr_f, args.weight_decay_f, args.lp_epochs, device)
        wandb.log({
            f"{name}_estp_test_acc_lm": estp_test_acc,
            f"{name}_best_val_acc_lm": best_val_acc,
        })
        if args.gnn_type != "":
            graph.ndata["feat"] = output
            output, cat_output = model.inference(graph, device, args.eval_batch_size)
            test_acc, estp_test_acc, best_val_acc = edge_classification_evaluation(graph, output, node_pairs, labels, eval_tag.split_idx, eval_tag.data_info["n_labels"], args.lr_f, args.weight_decay_f, args.lp_epochs, device)
            wandb.log({
                f"{name}_estp_test_acc_gnn": estp_test_acc,
                f"{name}_best_val_acc_gnn": best_val_acc,
            })
            test_acc, estp_test_acc, best_val_acc = edge_classification_evaluation(graph, cat_output, node_pairs, labels, eval_tag.split_idx, eval_tag.data_info["n_labels"], args.lr_f, args.weight_decay_f, args.lp_epochs, device)
            wandb.log({
                f"{name}_estp_test_acc_gnn_cat": estp_test_acc,
                f"{name}_best_val_acc_gnn_cat": best_val_acc,
            })
    else:
        graph = eval_tag.test_graph
        link_prediction_evaluation(graph, output)

    return output

def evaluate_mol(args, model, device, name=""):
    eval_mol = Mol(args, name)
    dataset = IterMolDataset(eval_mol, 0, args.batch_size)
    eval_loader = DataLoader(dataset, shuffle=False, batch_size=None)
    model.eval()
    pooler = pool(args.pooler)
    output_lm, output_gnn, output_gnn_cat = [], [], []
    labels = []

    if args.incontext_eval:
        acc = incontext_evaluate(args, None, output_lm, name)
        acc = incontext_evaluate(args, None, output_gnn, name)
        acc = incontext_evaluate(args, None, output_gnn_cat, name)

    evaluator = Evaluator(name='ogbg-molhiv' if name == "hiv" else 'ogbg-molpcba' if name == "pcba" else 'ogbg-molchembl')
    test_acc, estp_test_acc, best_val_acc = graph_classification_evaluation(output_lm, eval_mol.labels, eval_mol.split_idx, eval_mol.num_tasks, args.lr_f, args.weight_decay_f, args.lp_epochs, evaluator, device)
    wandb.log({
        "estp_test_acc_lm": estp_test_acc,
        "best_val_acc_lm": best_val_acc,
    })
    if args.gnn_type != "":
        test_acc, estp_test_acc, best_val_acc = graph_classification_evaluation(output_gnn, eval_mol.labels, eval_mol.split_idx, eval_mol.num_tasks, args.lr_f, args.weight_decay_f, args.lp_epochs, evaluator, device)
        if best_val_acc > evaluate_mol.g_val_acc_gnn:
            evaluate_mol.g_val_acc_gnn = best_val_acc
            evaluate_mol.g_test_acc_gnn = estp_test_acc
        wandb.log({
            "estp_test_acc_gnn": estp_test_acc,
            "best_val_acc_gnn": best_val_acc,
        })

def train(self):
        if args.run_name != "":
            wandb.init(project="unigraph", entity=args.run_entity, name=args.run_name)
        else:
            wandb.init(project="unigraph", entity=args.run_entity)
        wandb.config.update(args)

        set_random_seed(args.seed)

        tags_data = []
        datasets = []
        for i, tag_name in enumerate(args.datasets_name):
            if tag_name not in ["hiv", 'pcba', 'chemblpre']:
                tags_data.append(TAG(args, tag_name))
                datasets.append(IterTAGDataset(tags_data[i], i, batch_size, args.num_roots, args.length))
            else:
                tags_data.append(Mol(args, tag_name))
                datasets.append(IterMolDataset(tags_data[i], i, batch_size))
                
        
        dataset = CombinedDataset(datasets, batch_size=batch_size)
        dataloader = DataLoader(dataset, batch_size=None)
        model = UniGraph(args, dataset.mask_token_id)

        optimizer = create_optimizer(args.optimizer, model, lr, args.weight_decay)
        num_training_steps = num_epochs * len(dataloader)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )
        model.to(device)

        # emb = evaluate(model, "initial")
        if args.load_checkpoint:
            if args.checkpoint_path != "":
                model.load_state_dict(torch.load(args.checkpoint_path))
            emb = evaluate(model, "initial")

        # training loop
        latent_loss = torch.tensor(0)
        progress_bar = tqdm(range(num_training_steps))
        for epoch in range(num_epochs):
            count = 0
            # for batch, batch_nodes in dataloader:
            for batch, batch_item, idx in dataloader:
                #print(batch_item)
                #print(batch)
                model.train()
                batch = {k: v.to(device) for k, v in batch.items()}
                # masked_batch = {k: v.to(device) for k, v in masked_batch.items()}
                if args.gnn_type == "":
                    masked_input_ids = batch_item
                    masked_input_ids = masked_input_ids.to(device)
                    loss = model(batch, masked_input_ids)
                else:
                    if isinstance(batch_item, dgl.DGLGraph):
                        graph = batch_item
                    else:
                        batch_nodes = batch_item
                        batch_nodes=torch.tensor(batch_nodes,dtype=torch.int64)
                        #print(idx)
                        #print(batch_nodes.shape)
                        #print(batch_nodes)
                        graph = dgl.node_subgraph(tags_data[idx].graph, batch_nodes)
                        graph = preprocess(graph)

                    graph = graph.to(device)
                    drop_g1 = drop_edge(graph, args.drop_edge_rate)
                    drop_g2 = drop_edge(graph, args.drop_edge_rate)
                    loss, latent_loss = model(batch, graph, epoch=epoch, drop_g1=drop_g1, drop_g2=drop_g2)
                loss.backward()
                optimizer.step()
                # lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

                wandb.log({
                    "current_lr": get_current_lr(optimizer),
                    "pretrain_loss": loss.item(),
                    "latent_loss": latent_loss.item(),
                    'num_edges': graph.number_of_edges(),
                    'num_nodes': graph.number_of_nodes(),
                })
                progress_bar.set_description(f"# Epoch {epoch}, train_loss: {loss.item():.8f}")
                count += 1
                if count % args.eval_steps == 0:
                    emb = evaluate(model, f"{epoch}_{count}")
                    torch.save(model.state_dict(), f"./checkpoints/{wandb.run.name}_step_{epoch}_{count}.pt")
        emb = evaluate(model, f"final")
        torch.save(model.state_dict(), f"./checkpoints/{wandb.run.name}_step_final.pt")


if __name__ == "__main__":
    args = build_args()
    train(args)
