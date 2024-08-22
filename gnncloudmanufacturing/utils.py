import torch
import dgl
import numpy as np
from dgl.data import DGLDataset


ss_type = ('s', 'ss', 's')
os_type = ('o', 'os', 's')
so_type = ('s', 'so', 'o')


def graph_from_problem(problem, gamma=None):
    n_tasks = problem['n_tasks']
    n_operations = problem['n_operations']
    operation = problem['operation']
    dist = problem['dist']
    time_cost = problem['time_cost']
    op_cost = problem['op_cost']
    productivity = problem['productivity']

    operation_index = []
    for i in range(n_tasks):
        for j in range(n_operations):
            if operation[j, i] == 1:
                operation_index.append((i, j))
    operation_index = np.array(operation_index)
    
    adj_operation = np.zeros((operation_index.shape[0], operation_index.shape[0]))
    for i in range(n_tasks):
        col_i = operation[:, i]
        path = np.where(col_i > 0)[0]
        for j in range(len(path) - 1):
            u = operation_index.tolist().index([i, path[j]])
            v = operation_index.tolist().index([i, path[j+1]])
            adj_operation[u, v] = 1

    full_time_cost = np.tile(time_cost, (n_tasks, 1))
    full_time_cost = full_time_cost[operation.T.reshape(-1).astype(bool)]

    full_op_cost = np.tile(op_cost, (n_tasks, 1))
    full_op_cost = full_op_cost[operation.T.reshape(-1).astype(bool)]

    graph_data = {
        ss_type: np.where(dist > 0),
        os_type: np.where(full_op_cost < 999),
        so_type: np.where(full_op_cost < 999)[::-1],
        ('o', 'forward', 'o'): np.where(adj_operation > 0),
        ('o', 'backward', 'o'): np.where(adj_operation > 0)[::-1],
    }
    g = dgl.heterograph(graph_data)
    g = dgl.add_self_loop(g, etype='ss')

    g.ndata['feat'] = {
        'o': torch.ones(len(operation_index), 1),
        's': torch.FloatTensor(productivity[:, None])
    }
    g.ndata['operation_index'] = {
        'o': torch.LongTensor(operation_index),
    }
    u_idx, v_idx = g.edges(etype='os')
    serves_feat = np.array([
        full_op_cost[u_idx, v_idx],
        full_time_cost[u_idx, v_idx],
    ])
    g.edata['feat'] = {
        'os': torch.FloatTensor(serves_feat.T),
        'ss': torch.FloatTensor(dist[g.edges(etype='ss')][:, None]),
    }
    g.edata['_feat'] = {
        'os': torch.FloatTensor(serves_feat.T),
        'ss': torch.FloatTensor(dist[g.edges(etype='ss')][:, None]),
    }

    target = []
    for full_o, c in zip(*np.where(full_op_cost < 999)):
        t, o = operation_index[full_o]
        if gamma is not None:
            target.append(gamma[o, t, c])
        else:
            target.append(0)
    g.edata['target'] = {
        'os': torch.FloatTensor(target)[:, None],
    }
    return g


def gamma_from_target(target, graph, problem):
    target_mask = target[:, 0] == 1
    u, v = graph.edges(etype=os_type)
    u, v = u[target_mask], v[target_mask]
    u = graph.ndata['operation_index']['o'][u]
    gamma = np.zeros((problem['n_operations'], problem['n_tasks'], problem['n_cities']))
    for i in range(len(u)):
        operation, task, city = u[i, 1], u[i, 0], v[i]
        gamma[operation, task, city] = 1
    return gamma


def delta_from_gamma(problem, gamma):
    n_cities = problem['n_cities']
    n_operations = problem['n_operations']
    n_tasks = problem['n_tasks']

    delta = np.zeros((1, n_cities, n_cities, n_operations - 1, n_tasks))
    for t in range(n_tasks):
        o_iter, c_iter = np.where(gamma[:, t] == 1)
        for i in range(len(o_iter)-1):
            o = o_iter[i]
            c_u, c_v = c_iter[i], c_iter[i+1]
            delta[0, c_u, c_v, o, t] = 1
    return delta


class GraphDataset(DGLDataset):
    def __init__(self, graphs):
        super().__init__(name='custom_dataset')
        self.graphs = graphs
        self.ids = torch.arange(len(graphs))
    
    def __getitem__(self, idx):
        return self.graphs[idx], self.ids[idx]
    
    def __len__(self):
        return len(self.graphs)
