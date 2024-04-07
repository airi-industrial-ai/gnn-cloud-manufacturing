import numpy as np


class NaiveSolver:
    def __init__(self, dataset):
        self.problems = dataset
        self.total_cost = 0

    def solve_suboperaion(
        self, available_operations, cost_operations,
        trans_cost, operation_number,
        gamma, delta
    ):
        """
        Solve problem for one suboperation
        """
        sub_problem_data = []

        for i, stage in enumerate(available_operations):
            if i == 0:
                city = np.argmin(cost_operations[stage])
                sub_problem_data.append([city, cost_operations[stage, city]])
            else:
                cost_total = cost_operations[stage] + trans_cost[:,city,:]

                # Here we calculate the min value of matrix in axis services-cities
                service, city = np.unravel_index(np.argmin(cost_total),
                                                cost_total.shape)
                
                sub_problem_data.append([city, cost_total[service, city]])
                delta[service, previous_city, city, stage, operation_number] = 1

            previous_city = city
            gamma[stage, operation_number, city] = 1

        return (
            np.array(sub_problem_data)[:, 0],
            np.sum(np.array(sub_problem_data)[:, 1]),
            gamma, delta
        )

    def solve_problem(self, num_problem):
        """
        Solve operation
        """
        n_cities = self.problems[num_problem]["n_cities"]
        n_services = self.problems[num_problem]["n_services"]
        n_suboperations = self.problems[num_problem]["n_suboperations"]
        n_operations = self.problems[num_problem]["n_operations"]
        operations = self.problems[num_problem]["operations"]
        dist = self.problems[num_problem]["dist"]
        time_cost = self.problems[num_problem]["time_cost"]
        op_cost = self.problems[num_problem]["op_cost"]
        productivity = self.problems[num_problem]["productivity"]
        transportation_cost = self.problems[num_problem]["transportation_cost"]

        # Create cost matrices
        cost_operations = time_cost * op_cost / productivity
        trans_cost = dist[None, ...] * transportation_cost

        gamma = np.zeros((n_suboperations, n_operations, n_cities))
        delta = np.zeros((n_services, n_cities, n_cities,
                          n_suboperations, n_operations))

        problem_cost = 0
        problem_path = {}
        for n_operat in range(n_operations):
            available_operations = np.nonzero(operations[:, n_operat])[0]
            path, cost, gamma, delta = self.solve_suboperaion(
                available_operations, cost_operations,
                trans_cost, n_operat, gamma, delta
            )

            problem_cost += cost
            problem_path[f"suboperation_{n_operat}"] = path

        return {"path": problem_path, "cost": problem_cost}, gamma, delta
