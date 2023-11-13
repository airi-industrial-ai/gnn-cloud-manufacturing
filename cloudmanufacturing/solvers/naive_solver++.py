import numpy as np


class NaiveSolver:
    def __init__(self, dataset, n_bests=4, step_forward=3):
        self.problems = dataset
        self.total_cost = 0
        self.city = None
        self.n_best = n_bests
        self.step_forward = step_forward
        self.cost_operations = None
        self.trans_cost = None

    def look_forward(self, available_operations, city=None):
        city_list = []
        cost = 0
        for stage in available_operations:
            if city is None:
                city = np.random.choice(
                    np.argsort(self.cost_operations[stage])[: self.n_best]
                )
                cost += self.cost_operations[stage][city]
            else:
                cost_total = self.cost_operations[stage] + self.trans_cost[city]
                city = np.random.choice(np.argsort(cost_total)[: self.n_best])
                cost += cost_total[city]

            if len(available_operations[1:]) > 0:
                cost, city = self.look_forward(available_operations[1:], city)
            city_list.append(city)
            return cost, city

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
