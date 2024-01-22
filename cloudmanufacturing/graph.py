import torch
import dgl
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch.nn.functional as F

ss_type = ('s', 'ss', 's')
os_type = ('o', 'os', 's')
so_type = ('s', 'so', 'o')

def dglgraph_fixed(problem, gamma, oper_max=20):
    g = dglgraph(problem, gamma)
    ncolumns = g.ndata['feat']['o'].shape[1]
    g.ndata['feat'] = {'o': F.pad(g.ndata['feat']['o'], [0, oper_max - ncolumns])}
    return g


def dglgraph(problem, gamma):
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
    operation_index.append([n_tasks // 2, -1])
    operation_index.append([n_tasks // 2, n_operations])
    operation_index = np.array(operation_index)
    
    adj_operation = np.zeros((operation_index.shape[0], operation_index.shape[0]))
    for i in range(n_tasks):
        col_i = operation[:, i]
        path = np.where(col_i > 0)[0]
        for j in range(len(path) - 1):
            u = operation_index.tolist().index([i, path[j]])
            v = operation_index.tolist().index([i, path[j+1]])
            adj_operation[u, v] = 1
        adj_operation[-2, operation_index.tolist().index([i, path[0]])] = 1
        adj_operation[operation_index.tolist().index([i, path[-1]]), -1] = 1

    full_time_cost = np.tile(time_cost, (n_tasks, 1))
    full_time_cost = full_time_cost[operation.T.reshape(-1).astype(bool)]

    full_op_cost = np.tile(op_cost, (n_tasks, 1))
    full_op_cost = full_op_cost[operation.T.reshape(-1).astype(bool)]

    target = []
    for full_o, c in zip(*np.where(full_op_cost < 999)):
        t, o = operation_index[full_o]
        target.append(gamma[o, t, c])

    graph_data = {
        ss_type: np.where(dist > 0),
        os_type: np.where(full_op_cost < 999),
        so_type: np.where(full_op_cost < 999)[::-1],
        ('o', 'forward', 'o'): np.where(adj_operation > 0),
        ('o', 'backward', 'o'): np.where(adj_operation > 0)[::-1],
    }
    g = dgl.heterograph(graph_data)
    g = dgl.add_self_loop(g, etype='ss')

    op_feat = OneHotEncoder().fit_transform(
        operation_index[g.nodes('o').numpy()][:, [1]]
    ).toarray()
    g.ndata['feat'] = {
        'o': torch.FloatTensor(op_feat),
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
    g.edata['target'] = {
        'os': torch.FloatTensor(target)[:, None],
    }
    return g


def graph_gamma(graph, problem):
    target_mask = graph.edata['target'][os_type][:, 0] == 1
    u, v = graph.edges(etype=os_type, )
    u, v = u[target_mask], v[target_mask]
    u = graph.ndata['operation_index']['o'][u]
    gamma = np.zeros((problem['n_operations'], problem['n_tasks'], problem['n_cities']))
    for i in range(len(u)):
        operation, task, city = u[i, 1], u[i, 0], v[i]
        gamma[operation, task, city] = 1
    return gamma
