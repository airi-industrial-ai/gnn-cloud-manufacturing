from mip import BINARY, MINIMIZE, Model, xsum
from itertools import product
import numpy as np

def find_path(gamma):
    problem_info = []
    for task in range(gamma.shape[1]):
        path = np.array([])
        for sub_operation in range(len(gamma)):
            city = np.nonzero(gamma[sub_operation][task])[0]
            path = np.concatenate([path, city])
        problem_info.append(list(map(int, path)))
    return problem_info

def mip_solve(problem):
    n_operations = problem["n_operations"]
    n_suboperations = problem["n_suboperations"]
    n_cities = problem["n_cities"]
    n_services = problem["n_services"]
    operations = problem["operations"]
    dist = problem["dist"]
    time_cost = problem["time_cost"]
    op_cost = problem["op_cost"]
    productivity = problem["productivity"]
    transportation_cost = problem["transportation_cost"]

    gamma_shape = (n_suboperations, n_operations, n_cities)
    delta_shape = (n_services, n_cities, n_cities, n_suboperations, n_operations)

    model = Model(sense=MINIMIZE)
    model.verbose = 0
    gamma = model.add_var_tensor(shape=gamma_shape, name="gamma", var_type=BINARY)
    delta = model.add_var_tensor(shape=delta_shape, name="delta", var_type=BINARY)

    # For each suboperation in operation we should have 1 for one city if suboperation == 1
    # This constrain force to perform each suboperation
    for i, k in product(range(n_suboperations), range(n_operations)):
        # example: m += xsum(w[i]*x[i] for i in range(n)) <= c !!!
        model += xsum(gamma[i, k]) == operations[i, k]

    for i, k, m, m_ in product(
        range(n_suboperations),
        range(n_operations),
        range(n_cities),
        range(n_cities),
    ):
        # Equation constrains the logistic service to only one service
        # if any logistic service is required between two service points to fulfill two following sub-operations.
        seq = np.where(operations[i:, k] == 1)[0]
        if operations[i, k] and len(seq) > 1:
            if m != m_:
                model += gamma[i, k, m] + gamma[i + seq[1], k, m_] - 1 <= xsum(
                    delta[:, m, m_, i + seq[1], k]
                )
    # model += xsum(delta.flatten()) <= xsum(gamma.flatten())

    total_op_cost = xsum(
        ((time_cost * op_cost / productivity)[:, None, :] * gamma).flatten()
    )

    total_logistic_cost = xsum(
        ((transportation_cost * dist[None, ...])[..., None, None] * delta).flatten()
    )
    model.objective = total_op_cost + total_logistic_cost

    status = model.optimize(max_seconds=120)

    _delta = np.reshape(list(map(lambda x: x.x, delta.flatten().tolist())), delta_shape)
    _gamma = np.reshape(list(map(lambda x: x.x, gamma.flatten().tolist())), gamma_shape)

    return _delta, _gamma, status, model.objective_value
