import numpy as np


def objvalue(problem, gamma, delta):
    time_cost = problem["time_cost"]
    op_cost = problem["op_cost"]
    productivity = problem["productivity"]
    transportation_cost = problem["transportation_cost"]
    dist = problem["dist"]

    total_op_cost = np.sum(
        (time_cost * op_cost / productivity[None, :])[:, None, :] * gamma
    )
    total_logistic_cost = np.sum(
        (transportation_cost[:, None, None] * dist[None, ...])[..., None, None] * delta
    )
    return total_op_cost + total_logistic_cost


def construct_delta(problem, gamma):
    n_cities = problem["n_cities"]
    n_suboperations = problem["n_suboperations"]
    n_operations = problem["n_operations"]

    delta = np.zeros((1, n_cities, n_cities, n_suboperations - 1, n_operations))
    for t in range(n_operations):
        o_iter, c_iter = np.where(gamma[:, t] == 1)
        for i in range(len(o_iter) - 1):
            o = o_iter[i]
            c_u, c_v = c_iter[i], c_iter[i + 1]
            delta[0, c_u, c_v, o, t] = 1
    return delta
