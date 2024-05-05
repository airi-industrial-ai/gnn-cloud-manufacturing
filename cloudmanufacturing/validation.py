import numpy as np
from torch.nn import functional as F


def objvalue(problem, gamma, delta):
    time_cost = problem["time_cost"]
    op_cost = problem["op_cost"]
    productivity = problem["productivity"]
    transportation_cost = problem["transportation_cost"]
    dist = problem["dist"]

    total_op_cost = np.sum(
        (time_cost * op_cost / productivity)[:, None, :] * gamma
    )
    total_logistic_cost = np.sum(
        (transportation_cost * dist[None, ...])[..., None, None] * delta
    )
    return total_op_cost + total_logistic_cost


def construct_delta(problem, gamma):
    n_cities = problem["n_cities"]
    n_services = problem['n_services']
    n_suboperations = problem["n_suboperations"]
    n_operations = problem["n_operations"]

    delta = np.zeros((n_services, n_cities, n_cities, n_suboperations, n_operations))
    for t in range(n_operations):
        o_iter, c_iter = np.where(gamma[:, t] == 1)
        for i in range(len(o_iter) - 1):
            o = o_iter[i]
            c_u, c_v = c_iter[i], c_iter[i + 1]
            delta[0, c_u, c_v, o, t] = 1
    return delta

def construct_delta2(problem, gamma, graph, logits):
    n_cities = problem["n_cities"]
    n_services = problem['n_services']
    n_suboperations = problem["n_suboperations"]
    n_operations = problem["n_operations"]
    delta = np.zeros((n_services+1, n_cities, n_cities, n_suboperations, n_operations))

    operation_index = graph.ndata['operation_index']['o']

    for t in range(n_operations):
        o_iter, c_iter = np.where(gamma[:, t] == 1)
        for i in range(len(o_iter) - 1):
            o = o_iter[i+1]
            c_u, c_v = c_iter[i], c_iter[i + 1]

            identifier = np.where((operation_index[0] == t) & (operation_index[1] == o))[0]
            edge_idx = np.where(
                (graph.edges(etype="os")[0] == identifier) & (graph.edges(etype="os")[1] == c_v)
            )[0]

            delta[np.argmax(F.softmax(logits), axis=1)[edge_idx],
                    c_u, c_v, o_iter[i+1], t] = 1
    return delta[:-1,]