import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

from cloudmanufacturing.conv_layers import (AttnConvLayer, DotProductDecoder,
                                            ConvLayer, sample_so)
from cloudmanufacturing.graph import ss_type, os_type, so_type


class GNN_att(nn.Module):
    def __init__(self,s_shape_init, o_shape_init, os_shape_init,
                 ss_shape_init, out_dim, n_layers):
        super().__init__()

        os_shape = o_shape_init + os_shape_init
        ss_shape = s_shape_init + ss_shape_init

        convs = [AttnConvLayer(s_shape_init, o_shape_init,
                               os_shape, ss_shape, out_dim)]
        for _ in range(n_layers-1):
            os_shape = out_dim + os_shape_init
            ss_shape = out_dim + ss_shape_init
            convs.append(AttnConvLayer(out_dim, out_dim,
                                       os_shape, ss_shape, out_dim))
        self.convs = nn.ModuleList(convs)
        self.dec = DotProductDecoder()

    def forward(self, graph):
        s_feat = graph.ndata['feat']['s']
        o_feat = graph.ndata['feat']['o']
        s_hid, o_hid, delta_logits = self.convs[0](graph, s_feat, o_feat)
        for conv in self.convs[1:]:
            s_hid, o_hid, delta_logits = conv(graph, torch.relu(s_hid), torch.relu(o_hid))
        logits = self.dec(graph, s_hid, o_hid)
        return logits, delta_logits

    def predict(self, graph, problem):
        logits, delta_logits = self.forward(graph)
        s, o = sample_so(graph, logits)
        operation_index = graph.ndata['operation_index']['o'][o]
        gamma = np.zeros(
            (problem['n_suboperations'], problem['n_operations'], problem['n_cities'])
        )
        for i in range(len(operation_index)):
            operation, task, city = operation_index[i, 1], operation_index[i, 0], s[i]
            gamma[operation, task, city] = 1

        delta = np.zeros(
            (problem['n_services'], problem['n_cities'], problem['n_cities'],
             problem['n_suboperations'], problem['n_operations'])
        )
        for i in range(len(operation_index)-1):
            if operation_index[i][0] == operation_index[i+1][0]:
                edge_idx = graph.edge_ids(s[i],s[i+1],etype=ss_type)
                serv = torch.argmax(F.softmax(delta_logits[edge_idx], dim=0))
                delta[serv, s[i], s[i+1], operation_index[i+1][1], operation_index[i][0]] = 1
        return gamma, delta


class GNN(nn.Module):
    def __init__(self,s_shape_init, o_shape_init, os_shape_init,
                 ss_shape_init, out_dim, n_layers):
        super().__init__()

        os_shape = o_shape_init + os_shape_init
        ss_shape = s_shape_init + ss_shape_init

        convs = [ConvLayer(s_shape_init, o_shape_init,
                               os_shape, ss_shape, out_dim)]
        for _ in range(n_layers-1):
            os_shape = out_dim + os_shape_init
            ss_shape = out_dim + ss_shape_init
            convs.append(ConvLayer(out_dim, out_dim,
                                       os_shape, ss_shape, out_dim))
        self.convs = nn.ModuleList(convs)
        self.dec = DotProductDecoder()

    def forward(self, graph):
        s_feat = graph.ndata['feat']['s']
        o_feat = graph.ndata['feat']['o']
        s_hid, o_hid, delta_logits = self.convs[0](graph, s_feat, o_feat)
        for conv in self.convs[1:]:
            s_hid, o_hid, delta_logits = conv(graph, torch.relu(s_hid), torch.relu(o_hid))
        logits = self.dec(graph, s_hid, o_hid)
        return logits, delta_logits

    def predict(self, graph, problem):
        logits, delta_logits = self.forward(graph)
        s, o = sample_so(graph, logits)
        operation_index = graph.ndata['operation_index']['o'][o]
        gamma = np.zeros(
            (problem['n_suboperations'], problem['n_operations'], problem['n_cities'])
        )
        for i in range(len(operation_index)):
            operation, task, city = operation_index[i, 1], operation_index[i, 0], s[i]
            gamma[operation, task, city] = 1

        delta = np.zeros(
            (problem['n_services'], problem['n_cities'], problem['n_cities'],
             problem['n_suboperations'], problem['n_operations'])
        )
        for i in range(len(operation_index)-1):
            if operation_index[i][0] == operation_index[i+1][0]:
                edge_idx = graph.edge_ids(s[i],s[i+1],etype=ss_type)
                serv = torch.argmax(F.softmax(delta_logits[edge_idx], dim=0))
                delta[serv, s[i], s[i+1], operation_index[i+1][1], operation_index[i][0]] = 1
        return gamma, delta
