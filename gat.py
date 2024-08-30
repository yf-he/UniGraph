

import logging
import numpy as np

import torch
import torch.nn as nn

import dgl
import tqdm

from dgl.ops import edge_softmax
import dgl.function as fn
from dgl.utils import expand_as_pair

from utils.functions import create_activation, create_norm


class GAT(nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 nhead,
                 nhead_out,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 norm,
                 concat_out=False,
                 encoding=False,
                 shared_layers=False,
                 extra_input_fc=False,
                 extra_output_fc=True,
                 ):
        super(GAT, self).__init__()
        self.out_dim = out_dim
        self.num_heads = nhead
        self.num_heads_out = nhead_out
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.concat_out = concat_out

        last_activation = create_activation(activation) if encoding else None
        last_residual = (encoding and residual)
        last_norm = norm if encoding else None

        hidden_in = in_dim
        hidden_out = out_dim

        # if shared_layers and num_layers > 1:
        #     hidden_in = num_hidden * nhead
        #     hidden_out = num_hidden * nhead
        #     if encoding:
        #         assert extra_input_fc
        #     else:
        #         assert extra_output_fc

        if num_layers == 1:
            self.gat_layers.append(GATConv(
                hidden_in, hidden_out, nhead_out,
                feat_drop, attn_drop, negative_slope, last_residual, norm=last_norm, concat_out=concat_out))
        else:
            # input projection (no residual)
            self.gat_layers.append(GATConv(
                hidden_in, num_hidden, nhead,
                feat_drop, attn_drop, negative_slope, residual, create_activation(activation), norm=norm, concat_out=concat_out))
            # hidden layers

            # if shared_layers:
            #     for l in range(1, num_layers):
            #         self.gat_layers.append(self.gat_layers[-1])
            # else:
            for l in range(1, num_layers - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gat_layers.append(GATConv(
                    num_hidden * nhead, num_hidden, nhead,
                    feat_drop, attn_drop, negative_slope, residual, create_activation(activation), norm=norm, concat_out=concat_out))
            # output projection
            self.gat_layers.append(GATConv(
                num_hidden * nhead, hidden_out, nhead_out,
                feat_drop, attn_drop, negative_slope, last_residual, activation=last_activation, norm=last_norm, concat_out=concat_out))

        if extra_input_fc:
            self.input_fc = nn.Linear(in_dim, num_hidden * nhead)
        else:
            self.input_fc = None
        
        if extra_output_fc:
            self.head = nn.Linear(num_hidden * nhead, out_dim)
        else:
            self.head = None

    # def aggre_edges(self, g, x, ex):
    #     # Function to apply to each edge
    #     def edge_func(edges):
    #         # Return a dictionary with the edge feature
    #         return {'e_feat': edges.data['feat']}

    #     g.ndata['feat'] = x
    #     g.edata['feat'] = ex
    #     # Apply function to all edges
    #     g.apply_edges(edge_func)

    #     # Aggregate edge features at their destination nodes
    #     g.send_and_recv(g.edges(), lambda edges: {'accum_feat': torch.sum(edges.data['e_feat'], dim=0)}, 
    #                     lambda nodes: {'accumulated_feat': torch.sum(nodes.mailbox['accum_feat'], dim=1) + nodes.data['feat']})
        
    #     return g.ndata['accumulated_feat']
    def aggre_edges(self, g, x, ex):
        g.edata['he'] = ex
        g.update_all(fn.copy_e('he', 'm'), fn.sum('m', 'he_aggr'))
        return g.ndata['he_aggr'] + x

        
    def forward(self, g, inputs, return_hidden=False):
        #h = inputs
        if 'feat' in g.edata and False:
            #print("aggre_edges")
            h = self.aggre_edges(g, inputs, g.edata['feat'])
        else:
            h = inputs
        if self.input_fc is not None:
            h = self.input_fc(h)

        hidden_list = [h]
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h)
            hidden_list.append(h)
            # if self.is_pretraining and self.training and l < self.num_layers - 1:
            #     mask = torch.randperm(h.shape[0], device=h.device)[:int(0.15 * h.shape[0])]
            #     h[mask] = self.hidden_mask_token[l]
        # output projection
        # if self.head is not None:
        #     return self.head(h)
        # else:
        #     return h
        if return_hidden:
            return self.head(h), torch.cat([h for h in hidden_list], dim=-1)
        else:
            return self.head(h)

    def inference(self, g, device, batch_size=128, emb=True):
        """
        Inference with the GAT model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        #x = g.ndata["feat"]
        if 'feat' in g.edata:
            x = self.aggre_edges(g, g.ndata["feat"], g.edata['feat'])
        else:
            x = g.ndata["feat"]
        num_heads = self.num_heads
        num_heads_out = self.num_heads_out
        hidden_list = [x]
        for l, layer in enumerate(self.gat_layers):
            if l < self.num_layers - 1:
                y = torch.zeros(g.num_nodes(), self.num_hidden * num_heads if l != len(self.gat_layers) - 1 else self.num_classes)
            else:
                if emb == False:
                    y = torch.zeros(g.num_nodes(), self.num_hidden if l != len(self.gat_layers) - 1 else self.num_classes)
                else:
                    #print(self.out_dim)
                    y = torch.zeros(g.num_nodes(), self.out_dim * num_heads_out)
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.DataLoader(
                    g,
                    torch.arange(g.num_nodes()),
                    sampler,
                    batch_size=batch_size,
                    shuffle=False,
                    drop_last=False,
                    num_workers=0)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].int().to(device)
                h = x[input_nodes].to(device)
                if l < self.num_layers - 1:
                    h = layer(block, h)
                else:
                    h = layer(block, h)
                    #h = h.mean(1)
                    #h = h.log_softmax(dim=-1)

                if l == len(self.gat_layers) - 1 and (emb == False):
                    h = self.head(h)
                y[output_nodes] = h.cpu()
            x = y
            hidden_list.append(y)
        cat_y = torch.cat([h for h in hidden_list], dim=-1)
        return y, cat_y

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.is_pretraining = False
        self.head = nn.Linear(self.num_heads * self.out_dim, num_classes)



class GATConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True,
                 norm=None,
                 concat_out=True):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._concat_out = concat_out

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = None
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        # if norm is not None:
        #     self.norm = norm(num_heads * out_feats)
        # else:
        #     self.norm = None
    
        self.norm = norm
        if norm is not None:
            self.norm = create_norm(norm)(num_heads * out_feats)
        self.set_allow_zero_in_degree(False)

    def reset_parameters(self):
        """
        Description
        -----------
        Reinitialize learnable parameters.
        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, get_attention=False):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise RuntimeError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                # h_dst = self.feat_drop(feat[1])
                h_dst = feat[1]

                if not hasattr(self, 'fc_src'):
                    feat_src = self.fc(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats)
                else:
                    feat_src = self.fc_src(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats)
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    *src_prefix_shape, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]
                    dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]
            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            # e[e == 0] = -1e3
            # e = graph.edata.pop('e')
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']

            # bias
            if self.bias is not None:
                rst = rst + self.bias.view(
                    *((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)

            # residual
            if self.res_fc is not None:
                # Use -1 rather than self._num_heads to handle broadcasting
                resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1, self._out_feats)
                rst = rst + resval

            if self._concat_out:
                rst = rst.flatten(1)
            else:
                rst = torch.mean(rst, dim=1)

            if self.norm is not None:
                rst = self.norm(rst)

            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst
