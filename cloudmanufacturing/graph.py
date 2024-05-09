import torch
import dgl
import numpy as np
from sklearn.preprocessing import OneHotEncoder

ss_type = ('s', 'ss', 's')
os_type = ('o', 'os', 's')
so_type = ('s', 'so', 'o')

def find_pairs(gamma):
    pairs_new = []
    indices = np.where(gamma.swapaxes(0, 1) == 1)
    for (t1, o1, c1), (t2, o2, c2) in zip(zip(*indices), 
                                        zip(*[index[1:] for index in indices])):
        if t1 == t2:
            pairs_new.append([c1, c2])
    return np.unique(np.stack(pairs_new, axis=0), axis=0)

def find_mask(pairs, dist):
    mask = (
        np.stack(np.where(dist>0)).T == np.array(pairs)[:,None]
    ).all(2).any(0)
    return np.append(mask,[False]*len(dist))

def create_dtarget(delta, dist, n_services):
    delta_target = np.zeros((dist.shape[0]**2, n_services))
    for i, (c1, c2) in enumerate(zip(*np.where(dist>0))):
        if len(np.nonzero(delta[:,c1,c2,:,:])[0]) > 0:
            delta_target[i, np.nonzero(delta[:,c1,c2,:,:])[0][0]] = 1
    return delta_target

def dglgraph(problem, gamma, delta):
    n_operations = problem['n_operations']
    n_suboperations = problem['n_suboperations']
    operations = problem['operations']
    dist = problem['dist']
    time_cost = problem['time_cost']
    op_cost = problem['op_cost']
    productivity = problem['productivity']
    transportation_cost = problem['transportation_cost']
    n_services = problem['n_services']

    operation_index = []
    for i in range(n_operations):
        for j in range(n_suboperations):
            if operations[j, i] == 1:
                operation_index.append((i, j))
    operation_index.append([n_operations // 2, -1])
    operation_index.append([n_operations // 2, n_suboperations])
    operation_index = np.array(operation_index)
    
    adj_operation = np.zeros((operation_index.shape[0], operation_index.shape[0]))
    for i in range(n_operations):
        col_i = operations[:, i]
        path = np.where(col_i > 0)[0]
        for j in range(len(path) - 1):
            u = operation_index.tolist().index([i, path[j]])
            v = operation_index.tolist().index([i, path[j+1]])
            adj_operation[u, v] = 1
        adj_operation[-2, operation_index.tolist().index([i, path[0]])] = 1
        adj_operation[operation_index.tolist().index([i, path[-1]]), -1] = 1

    full_time_cost = np.tile(time_cost, (n_operations, 1))
    full_time_cost = full_time_cost[operations.T.reshape(-1).astype(bool)]

    full_op_cost = np.tile(op_cost, (n_operations, 1))
    full_op_cost = full_op_cost[operations.T.reshape(-1).astype(bool)]

    target = []
    for full_o, c in zip(*np.where(full_op_cost < 999)):
        t, o = operation_index[full_o]
        target.append(gamma[o, t, c])

    pairs = find_pairs(gamma)
    mask = find_mask(pairs, dist)
    delta_target = create_dtarget(delta, dist, n_services)

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
        's': torch.FloatTensor(productivity)
    }
    g.ndata['operation_index'] = {
        'o': torch.LongTensor(operation_index),
    }
    u_idx, v_idx = g.edges(etype='os')
    serves_feat = np.array([
        full_op_cost[u_idx, v_idx],
        full_time_cost[u_idx, v_idx],
    ])

    transp = transportation_cost[:,
                             g.edges(etype="ss")[0],
                             g.edges(etype="ss")[1]] * dist[g.edges(etype="ss")]
    
    g.edata['feat'] = {
        'os': torch.FloatTensor(serves_feat.T),
        'ss': torch.FloatTensor(transp.T),
    }
    g.edata['target'] = {
        'os': torch.FloatTensor(target)[:, None],
    }
    g.edata['delta_target'] = {
        'ss': torch.FloatTensor(delta_target),
    }
    return g, mask

def graph_gamma(graph, problem):
    target_mask = graph.edata['target'][os_type][:, 0] == 1
    u, v = graph.edges(etype=os_type, )
    u, v = u[target_mask], v[target_mask]
    u = graph.ndata['operation_index']['o'][u]
    gamma = np.zeros((problem['n_suboperations'], problem['n_operations'], problem['n_cities']))
    for i in range(len(u)):
        operation, task, city = u[i, 1], u[i, 0], v[i]
        gamma[operation, task, city] = 1
    return gamma