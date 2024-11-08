import numpy as np
import pandas as pd
from typing import Dict
from dataclasses import dataclass

@dataclass
class Problem:
    operations_matrix: pd.DataFrame
    distances_matrix: pd.DataFrame
    times_matrix: pd.DataFrame
    costs_matrix: pd.DataFrame
    productivity: pd.Series
    distances_coef: float

class DPSolver:

    def solve(self, problem: Problem):
        
        solution = {}

        operations = list(problem.operations_matrix.columns)
        cities = list(problem.distances_matrix.index)

        ls_matrix = problem.distances_matrix.values * problem.distances_coef
        os_matrix = (problem.costs_matrix.values * problem.times_matrix.values) / problem.productivity.values

        os_matrix[os_matrix==np.inf] = 2**32
        ls_matrix[ls_matrix==np.inf] = 2**32

        for operation in operations:

            costs = {}
            next_cities = {}

            sub_operations_mask = (problem.operations_matrix[operation]==1).values

            sub_operations = list(problem.operations_matrix[sub_operations_mask].index)

            os_matrix_masked = os_matrix[sub_operations_mask]

            i = len(sub_operations) - 1

            for m, city in enumerate(cities):
                costs[(sub_operations[i], city)] = os_matrix_masked[i, m]

            while i > 0:
                i -= 1

                for m, city in enumerate(cities):
                     
                    next_m_costs = [os_matrix_masked[i, m] + ls_matrix[m, next_m] + costs[(sub_operations[i+1], next_city)]
                                    for next_m, next_city in enumerate(cities)]
                    
                    costs[(sub_operations[i], city)] = min(next_m_costs)

                    next_cities[(sub_operations[i], city)] = cities[np.argmin(next_m_costs)]

            operation_solution = {}

            prev_city = None

            for sub_operation in sub_operations:

                if prev_city is None:
                    prev_city = min([(city, costs[(sub_operation, city)]) for city in cities], key=lambda x: x[1])[0]
                    operation_solution[sub_operation] = prev_city
                    prev_sub_operation = sub_operation
                else:
                    prev_city = next_cities[(prev_sub_operation, prev_city)]
                    operation_solution[sub_operation] = prev_city
                    prev_sub_operation = sub_operation

            solution[operation] = operation_solution

        return solution

def dp_solve(problem):
    n_cities = problem["n_cities"]
    n_operations = problem["n_operations"]
    n_tasks = problem["n_tasks"]
    operations = problem["operation"]
    dist = problem["dist"]
    time_cost = problem["time_cost"]
    op_cost = problem["op_cost"]
    productivity = problem["productivity"]
    transportation_cost = problem["transportation_cost"]

    operations_matrix = pd.DataFrame(operations)
    operations_matrix.index = [f'Sub-operation{i}' for i in range(operations.shape[0])]
    operations_matrix.columns = [f'Operation{i}' for i in range(operations.shape[1])]

    distances_matrix = pd.DataFrame(dist)
    distances_matrix.index = [f'city{i}' for i in range(dist.shape[0])]
    distances_matrix.columns = [f'city{i}' for i in range(dist.shape[1])]

    times_matrix = pd.DataFrame(time_cost)
    times_matrix.index = [f'Sub-operation{i}' for i in range(time_cost.shape[0])]
    times_matrix.columns = [f'city{i}' for i in range(time_cost.shape[1])]
    
    costs_matrix = pd.DataFrame(op_cost)
    costs_matrix.index = [f'Sub-operation{i}' for i in range(op_cost.shape[0])]
    costs_matrix.columns = [f'city{i}' for i in range(op_cost.shape[1])]

    productivity = pd.Series(productivity)
    productivity.index = [f'city{i}' for i in range(productivity.shape[0])]
    
    distances_coef = transportation_cost[0]
    
    aux_problem = Problem(
        operations_matrix=operations_matrix,
        distances_matrix=distances_matrix,
        times_matrix=times_matrix,
        costs_matrix=costs_matrix,
        productivity=productivity,
        distances_coef=distances_coef,
    )
    
    solver = DPSolver()
    problem_solution = solver.solve(aux_problem)

    gamma = np.zeros((n_operations, n_tasks, n_cities))
    for t in problem_solution:
        for o in problem_solution[t]:
            c = problem_solution[t][o]
            i = int(t[9:]) - 1
            j = int(o[13:]) - 1
            k = int(c[4:]) - 1
            gamma[j, i, k] = 1
    
    return gamma
