import numpy as np

class naive_solver():
    def __init__(self, dataset):
        self.problems = dataset
    

    def solve_sub_problem(self, available_operations, cost_operations, trans_cost):
        sub_problem_data = []
        for i, stage in enumerate(available_operations):
            if i == 0:
                city = np.argmin(cost_operations[stage])
                sub_problem_data.append([city, cost_operations[stage][city]])
            else:
                cost_total = cost_operations[stage] + trans_cost[city]
                city = np.argmin(cost_total)
                sub_problem_data.append([city, cost_operations[stage][city]])        
        return np.array(sub_problem_data)[:,0], np.sum(np.array(sub_problem_data)[:,1])

    

    def solve_problem(self, num_problem):
        # Extract data
        n_tasks = self.problems[num_problem]['n_tasks']
        n_operations = self.problems[num_problem]['n_operations']
        n_cities = self.problems[num_problem]['n_cities']
        n_services = self.problems[num_problem]['n_services']
        operation = self.problems[num_problem]['operation']
        dist = self.problems[num_problem]['dist']
        time_cost = self.problems[num_problem]['time_cost']
        op_cost = self.problems[num_problem]['op_cost']
        productivity = self.problems[num_problem]['productivity']
        transportation_cost = self.problems[num_problem]['transportation_cost']

        # Create cost matrices
        cost_operations = time_cost * op_cost / productivity
        trans_cost = dist * transportation_cost

        total_cost = 0
        total_path = {}
        for task_number in range(n_tasks):
            available_operations = np.nonzero(operation[:,task_number])[0]
            path, cost = self.solve_sub_problem(available_operations, cost_operations, trans_cost)
            
            total_cost += cost
            total_path[f'subproblem_{task_number}'] = path
        
        return total_path, total_cost