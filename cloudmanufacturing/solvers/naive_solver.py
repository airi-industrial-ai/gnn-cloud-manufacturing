import numpy as np


class NaiveSolver:
    def __init__(self, dataset):
        self.problems = dataset
        self.total_cost = 0
        self.city = None

    def solve_suboperaion(self, available_operations, cost_operations, trans_cost):
        """
        Solve problem for one suboperation
        """
        sub_problem_data = []
        for stage in available_operations:
            if self.city is None:
                self.city = np.argmin(cost_operations[stage])
                sub_problem_data.append([self.city, cost_operations[stage][self.city]])
            else:
                cost_total = cost_operations[stage] + trans_cost[self.city]
                self.city = np.argmin(cost_total)
                sub_problem_data.append([self.city, cost_total[self.city]])
        return np.array(sub_problem_data)[:, 0], np.sum(
            np.array(sub_problem_data)[:, 1]
        )

    def solve_problem(self, num_problem, save_costs=False):
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
                available_operations, cost_operations, trans_cost
            )

            problem_cost += cost
            problem_path[f"suboperation_{n_sub}"] = path
        if save_costs:
            self.total_cost += problem_cost
        self.city = None
        return {"path": problem_path, "cost": problem_cost}

    def solve_all(self):
        self.total_cost = 0
        problem_info = {}
        for num in range(len(self.problems)):
            result = self.solve_problem(num, save_costs=True)
            problem_dict = {}
            problem_dict["path"] = result["path"]
            problem_dict["cost"] = result["cost"]
            problem_info[f"problem_{num}"] = problem_dict

        return problem_info
