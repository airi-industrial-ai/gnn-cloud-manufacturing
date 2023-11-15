import numpy as np


class naive_solver:
    def __init__(self, dataset):
        self.problems = dataset
        self.total_cost = 0

        n_tasks = dataset[0]["n_tasks"]
        n_operations = dataset[0]["n_operations"]
        n_cities = dataset[0]["n_cities"]
        self.gamma = np.zeros((n_operations, n_tasks, n_cities))

    def solve_suboperaion(
        self,
        available_operations,
        cost_operations,
        trans_cost,
        operation_number=None,
        save_to_gamma=False,
    ):
        """
        Solve problem for one suboperation
        """
        sub_problem_data = []

        for i, stage in enumerate(available_operations):
            if i == 0:
                city = np.argmin(cost_operations[stage])
                sub_problem_data.append([city, cost_operations[stage][city]])
            else:
                cost_total = cost_operations[stage] + trans_cost[city]
                city = np.argmin(cost_total)
                sub_problem_data.append([city, cost_operations[stage][city]])

            if save_to_gamma is True:
                self.gamma[operation_number, stage, city] = 1

        return np.array(sub_problem_data)[:, 0], np.sum(
            np.array(sub_problem_data)[:, 1]
        )

    def solve_problem(self, num_problem, save_costs=False, save_to_gamma=True):
        """
        Solve operation
        """
        n_tasks = self.problems[num_problem]["n_tasks"]
        operation = self.problems[num_problem]["operation"]
        dist = self.problems[num_problem]["dist"]
        time_cost = self.problems[num_problem]["time_cost"]
        op_cost = self.problems[num_problem]["op_cost"]
        productivity = self.problems[num_problem]["productivity"]
        transportation_cost = self.problems[num_problem]["transportation_cost"]

        # Create cost matrices
        cost_operations = time_cost * op_cost / productivity
        trans_cost = dist * transportation_cost

        problem_cost = 0
        problem_path = {}
        for n_sub in range(n_tasks):
            available_operations = np.nonzero(operation[:, n_sub])[0]
            path, cost = self.solve_suboperaion(
                available_operations,
                cost_operations,
                trans_cost,
                operation_number=n_sub,
                save_to_gamma=save_to_gamma,
            )

            problem_cost += cost
            problem_path[f"suboperation_{n_sub}"] = path
        if save_costs:
            self.total_cost += problem_cost
        return {"path": problem_path, "cost": problem_cost}

    def solve_all(self):
        self.total_cost = 0
        problem_info = {
            f"problem_{num}": {
                "path": self.solve_problem(num, save_costs=True, save_to_gamma=True)[
                    "path"
                ],
                "cost": self.solve_problem(num, save_costs=True, save_to_gamma=True)[
                    "cost"
                ],
            }
            for num in range(len(self.problems))
        }

        return problem_info
