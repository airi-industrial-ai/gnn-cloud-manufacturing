import numpy as np
from gnncloudmanufacturing.utils import os_type, ss_type
from itertools import product


def total_cost_from_gamma(problem, gamma, delta):
    time_cost = problem['time_cost']
    op_cost = problem['op_cost']
    productivity = problem['productivity']
    transportation_cost = problem['transportation_cost']
    dist = problem['dist']

    total_op_cost = np.sum(
        (time_cost * op_cost / productivity[None, :])[:, None, :] * gamma
    )
    total_logistic_cost = np.sum(
        (transportation_cost[:, None, None] * dist[None, ...])[..., None, None] * delta
    )
    return total_op_cost + total_logistic_cost


def total_cost_from_graph(graph, pred, transportation_cost=0.3):
    mask = pred.bool()[:, 0]
    o, s = graph.edges(etype=os_type)
    o, s = o[mask], s[mask]
    edata_feat = graph.edata['_feat'][os_type][mask]
    productivity = graph.dstdata['feat']['s'][s][:, 0]
    op_cost = edata_feat[:, 0]
    time_cost = edata_feat[:, 1]
    total_op_cost = sum(time_cost * op_cost / productivity)

    total_logistic_cost = 0
    for task in set(graph.ndata['operation_index']['o'][o, 0].numpy()):
        route = s[graph.ndata['operation_index']['o'][o, 0] == task]
        route_ids = graph.edge_ids(route[:-1], route[1:], etype=ss_type)
        dist = graph.edata['_feat'][ss_type][route_ids]
        total_logistic_cost += (dist * transportation_cost).sum()
    
    return total_op_cost + total_logistic_cost


def check_feasibility(gamma, delta, problem):
    n_operations = problem['n_operations']
    n_tasks = problem['n_tasks']
    operation = problem['operation']
    n_cities = problem['n_cities']
    for i, k in product(range(n_operations), range(n_tasks)):
        assert sum(gamma[i, k]) == operation[i, k]
    for i, k, m, m_ in product(
        range(n_operations-1), range(n_tasks), range(n_cities), range(n_cities)):
        seq = np.where(operation[i:, k] == 1)[0]
        if operation[i, k] and len(seq) > 1:
            assert gamma[i, k, m] + gamma[i+seq[1], k, m_] - 1 <= sum(delta[:, m, m_, i, k])
