import numpy as np
from cloudmanufacturing.validation import objvalue
from cloudmanufacturing.solvers.naive_solver import NaiveSolver

def validate_optimal(dataset, name):
    objvalue_score = []
    for i in range(len(dataset.problems)):
        gamma = dataset.gammas[i]
        delta = dataset.deltas[i]
        objvalue_score.append(
            objvalue(dataset.problems[i], gamma, delta)
        )
    return {f'objective {name}': np.mean(objvalue_score)}

def validate_greedy(dataset, name):
    problems = [problem for problem in dataset.problems]
    naive_solver = NaiveSolver(problems)
    objvalue_score = []
    for i, problem in enumerate(problems):
        _, gamma, delta = naive_solver.solve_problem(i)
        objvalue_score.append(objvalue(problem, gamma, delta))
    return {f'objective {name}': np.mean(objvalue_score)}

def validate_objective(model, dataset, name):
    objvalue_score = []
    for i in range(len(dataset.problems)):
        pred_gamma, pred_delta = model.predict(dataset.__getitem__(i), dataset.problems[i])
        objvalue_score.append(
            objvalue(dataset.problems[i], pred_gamma, pred_delta)
        )
    return {f'objective {name}': np.mean(objvalue_score)}