import dgl
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder

ss_type = ("s", "ss", "s")
os_type = ("o", "os", "s")
so_type = ("s", "so", "o")


def dglgraph(problem, gamma, delta):
    n_operations = problem["n_operations"]
    n_suboperations = problem["n_suboperations"]
    operations = problem["operations"]
    dist = problem["dist"]
    time_cost = problem["time_cost"]
    op_cost = problem["op_cost"]
    productivity = problem["productivity"]
    transportation_cost = problem["transportation_cost"]

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
            v = operation_index.tolist().index([i, path[j + 1]])
            adj_operation[u, v] = 1
        adj_operation[-2, operation_index.tolist().index([i, path[0]])] = 1
        adj_operation[operation_index.tolist().index([i, path[-1]]), -1] = 1

    full_time_cost = np.tile(time_cost, (n_operations, 1))
    full_time_cost = full_time_cost[operations.T.reshape(-1).astype(bool)]

    full_op_cost = np.tile(op_cost, (n_operations, 1))
    full_op_cost = full_op_cost[operations.T.reshape(-1).astype(bool)]

    target = []

    # take all operations and cities
    # for i, (full_o, c) in enumerate(zip(*np.where(full_op_cost < 999))):
    #     # split operations by suboperation and operation
    #     t, o = operation_index[full_o]
    #     # Take next suboperations which are not 0
    #     seq = np.where(operations[o:, t] == 1)[0]
    #     if len(seq) > 1:
    #         fc = np.nonzero(gamma[o, t, :])[0][0]
    #         sc = np.nonzero(gamma[o+seq[1], t, :])[0][0]
    #         if gamma[o, t, c] and any(delta[:, fc, sc, o+seq[1], t]):
    #             # find a company number
    #             company = np.nonzero(delta[:, fc, sc, o+seq[1], t])[0][0]
    #             target.append(gamma[o, t, c] + company)
    #     else:
    #         target.append(gamma[o, t, c])
    
    for i, (full_o, c) in enumerate(zip(*np.where(full_op_cost < 999))):
        # split operations by suboperation and operation
        t, o = operation_index[full_o]
        seq = np.where(operations[o:, t] == 1)[0]
        if len(seq) > 1:
            next_city = np.nonzero(gamma[o+seq[1], t, :])[0][0]
            next_oper = operation_index[full_o+1][1]
            if any(delta[:, c, next_city, next_oper, t]):
                company = np.nonzero(delta[:, c, next_city, next_oper, t])[0][0] + 1
            else: company = 0
        target.append(gamma[o, t, c] + company)
    target = torch.FloatTensor(target)
    target[target > 1] = target[target > 1] - 1

    graph_data = {
        ss_type: np.where(dist > 0),
        os_type: np.where(full_op_cost < 999),
        so_type: np.where(full_op_cost < 999)[::-1],
        ("o", "forward", "o"): np.where(adj_operation > 0),
        ("o", "backward", "o"): np.where(adj_operation > 0)[::-1],
    }
    g = dgl.heterograph(graph_data)
    g = dgl.add_self_loop(g, etype="ss")

    op_feat = (
        OneHotEncoder()
        .fit_transform(operation_index[g.nodes("o").numpy()][:, [1]])
        .toarray()
    )
    g.ndata["feat"] = {
        "o": torch.FloatTensor(op_feat),
        "s": torch.FloatTensor(productivity[:, None]),
    }
    g.ndata["operation_index"] = {
        "o": torch.LongTensor(operation_index),
    }
    u_idx, v_idx = g.edges(etype="os")
    serves_feat = np.array(
        [
            full_op_cost[u_idx, v_idx],
            full_time_cost[u_idx, v_idx],
        ]
    )
    transp = np.array(transportation_cost[:,
                             g.edges(etype="ss")[0],
                             g.edges(etype="ss")[1]] * dist[g.edges(etype="ss")])
    g.edata["feat"] = {
        "os": torch.FloatTensor(serves_feat.T),
        "ss": torch.FloatTensor(transp.T),
    }
    g.edata["target"] = {
        "os": torch.FloatTensor(target),
    }
    return g


def graph_gamma(graph, problem):
    # Take non-zero edges
    target_mask = graph.edata["target"][os_type] != 0
    
    # Return source and destination nodes
    u, v = graph.edges(
        etype=os_type,
    )
    # Apply mask
    u, v = u[target_mask], v[target_mask]
    # take operations
    u = graph.ndata["operation_index"]["o"][u]
    # Construct a new gamma matrix
    gamma = np.zeros(
        (problem["n_suboperations"], problem["n_operations"], problem["n_cities"])
    )
    # Fill the new gamma matrix
    for i in range(len(u)):
        operation, task, city = u[i, 1], u[i, 0], v[i]
        gamma[operation, task, city] = 1
    return gamma
