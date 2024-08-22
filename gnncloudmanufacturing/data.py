import numpy as np
import numpy.random as random
import pandas as pd
from tqdm.auto import tqdm

def read_fatahi_dataset(path_to_file, sheet_names=None):
    """
    Fatahi Valilai, Omid. “Dataset for Logistics and Manufacturing 
    Service Composition”. 17 Mar. 2021. Web. 9 June 2023.
    """
    if sheet_names is None:
        sheet_names = [
            '5,10,10-1',
            '5,10,10-2',
            '5,10,10-3',
            '10,10,10-1',
            '10,10,10-2',
            '10,10,10-3',
            '5,10,20-1',
            '5,10,20-2',
            '5,10,20-3',
            '5,20,10-1',
            '5,20,10-2',
            '5,20,10-3',
            '5,20,20-1',
            '5,20,20-2',
            '5,20,20-3',
            '5,5,5-1',
            '5,5,5-2',
            '5,5,5-3',
        ]
    res = []
    for sheet_name in tqdm(sheet_names):
        res.append(_read_sheet(path_to_file, sheet_name))
    return res


def _read_sheet(path_to_file, sheet_name):
    n_services = 1
    n_tasks, n_operations, n_cities, _ = list(
            map(int, '-'.join(sheet_name.split(',')).split('-'))
        )
    operation = np.zeros((n_operations, n_tasks))
    dist = np.zeros((n_cities, n_cities))
    time_cost = np.zeros((n_operations, n_cities))
    op_cost = np.zeros((n_operations, n_cities))
    productivity = np.zeros((n_cities))
    transportation_cost = np.zeros((n_services))

    operation[:, :] = pd.read_excel(
        path_to_file, 
        sheet_name=sheet_name, 
        header=None, 
        usecols=range(1, n_tasks+1),
        skiprows=5,
        nrows=n_operations,
    )
    dist[:, :] = pd.read_excel(
        path_to_file, 
        sheet_name=sheet_name, 
        header=None, 
        usecols=range(1, n_cities+1), 
        skiprows=5*2+n_operations-1,
        nrows=n_cities,
    )
    time_cost[:, :] = pd.read_excel(
        path_to_file, 
        sheet_name=sheet_name, 
        header=None, 
        usecols=range(1, n_cities+1), 
        skiprows=5*3+n_operations+n_cities-1*2,
        nrows=n_operations,
    )
    time_cost[np.isinf(time_cost)] = 999
    op_cost[:, :] = pd.read_excel(
        path_to_file, 
        sheet_name=sheet_name, 
        header=None, 
        usecols=range(1, n_cities+1), 
        skiprows=5*4+n_operations+n_cities+n_operations-1*3,
        nrows=n_operations,
    )
    op_cost[np.isinf(op_cost)] = 999
    productivity[:] = pd.read_excel(
        path_to_file, 
        sheet_name=sheet_name, 
        header=None, 
        usecols=range(n_cities), 
        skiprows=5*5+n_operations+n_cities+n_operations+n_operations-1*4,
        nrows=1,
    )
    transportation_cost[:] = [0.3]
    return {
        'name': sheet_name,
        'n_tasks': n_tasks, 
        'n_operations': n_operations, 
        'n_cities': n_cities,
        'n_services': n_services,
        'operation': operation,
        'dist': dist,
        'time_cost': time_cost,
        'op_cost': op_cost,
        'productivity': productivity,
        'transportation_cost': transportation_cost,
    }


def sample_problem(
        n_tasks, 
        n_operations, 
        n_cities, 
        threshold=0.5, 
        max_iters=1000, 
        dirpath='../data/',
        random_seed=None):
    assert 0 < n_tasks
    assert 0 < n_operations < 21
    assert 0 < n_cities < 21
    if random_seed is not None:
        random.seed(random_seed)
    for i in range(max_iters):
        operation = random.rand(n_operations, n_tasks) > threshold
        operation = operation.astype(int)
        if np.all(operation.sum(axis=0) > 0):
            break
    assert np.all(operation.sum(axis=0) > 0)
    dist = np.load(f'{dirpath}dist.npy')[:n_cities, :n_cities]
    time_cost = np.load(f'{dirpath}time_cost.npy')[:n_operations, :n_tasks]
    op_cost = np.load(f'{dirpath}op_cost.npy')[:n_operations, :n_tasks]
    productivity = np.load(f'{dirpath}productivity.npy')[:n_cities]
    transportation_cost = np.array([0.3])
    return {
        'name': f'{n_tasks},{n_operations},{n_cities}',
        'n_tasks': n_tasks, 
        'n_operations': n_operations, 
        'n_cities': n_cities,
        'n_services': 1,
        'operation': operation,
        'dist': dist,
        'time_cost': time_cost,
        'op_cost': op_cost,
        'productivity': productivity,
        'transportation_cost': transportation_cost,
    }


def sample_dataset(
        n_problems, 
        n_tasks_range=[5,10], 
        n_operations_range=[5,20], 
        n_cities_range=[5,20], 
        threshold=0.5, 
        max_iters=1000, 
        dirpath='../data/',
        random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)
    problems = []
    for i in range(n_problems):
        n_tasks = random.randint(n_tasks_range[0], n_tasks_range[1]+1)
        n_operations = random.randint(n_operations_range[0], n_operations_range[1]+1)
        n_cities = random.randint(n_cities_range[0], n_cities_range[1]+1)
        problem = sample_problem(n_tasks, n_operations, n_cities, threshold, max_iters, dirpath)
        problems.append(problem)
    return problems
