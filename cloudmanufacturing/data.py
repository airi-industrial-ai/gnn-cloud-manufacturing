import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def read_fatahi_dataset(path_to_file):
    """
    Fatahi Valilai, Omid. “Dataset for Logistics and Manufacturing
    Service Composition”. 17 Mar. 2021. Web. 9 June 2023.
    """
    sheet_names = [
        "5,10,10-1",
        "5,10,10-2",
        "5,10,10-3",
        "10,10,10-1",
        "10,10,10-2",
        "10,10,10-3",
        "5,10,20-1",
        "5,10,20-2",
        "5,10,20-3",
        "5,20,10-1",
        "5,20,10-2",
        "5,20,10-3",
        "5,20,20-1",
        "5,20,20-2",
        "5,20,20-3",
        "5,5,5-1",
        "5,5,5-2",
        "5,5,5-3",
    ]
    res = []
    for sheet_name in tqdm(sheet_names):
        res.append(_read_sheet(path_to_file, sheet_name))
    return res


def _read_sheet(path_to_file, sheet_name):
    n_services = 1
    n_tasks, n_operations, n_cities, _ = list(
        map(int, "-".join(sheet_name.split(",")).split("-"))
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
        usecols=range(1, n_tasks + 1),
        skiprows=5,
        nrows=n_operations,
    )
    dist[:, :] = pd.read_excel(
        path_to_file,
        sheet_name=sheet_name,
        header=None,
        usecols=range(1, n_cities + 1),
        skiprows=5 * 2 + n_operations - 1,
        nrows=n_cities,
    )
    time_cost[:, :] = pd.read_excel(
        path_to_file,
        sheet_name=sheet_name,
        header=None,
        usecols=range(1, n_cities + 1),
        skiprows=5 * 3 + n_operations + n_cities - 1 * 2,
        nrows=n_operations,
    )
    time_cost[np.isinf(time_cost)] = 99
    op_cost[:, :] = pd.read_excel(
        path_to_file,
        sheet_name=sheet_name,
        header=None,
        usecols=range(1, n_cities + 1),
        skiprows=5 * 4 + n_operations + n_cities + n_operations - 1 * 3,
        nrows=n_operations,
    )
    op_cost[np.isinf(op_cost)] = 999
    productivity[:] = pd.read_excel(
        path_to_file,
        sheet_name=sheet_name,
        header=None,
        usecols=range(n_cities),
        skiprows=5 * 5 + n_operations + n_cities + n_operations + n_operations - 1 * 4,
        nrows=1,
    )
    transportation_cost[:] = [0.3]
    return {
        "name": sheet_name,
        "n_tasks": n_tasks,
        "n_operations": n_operations,
        "n_cities": n_cities,
        "n_services": n_services,
        "operation": operation,
        "dist": dist,
        "time_cost": time_cost,
        "op_cost": op_cost,
        "productivity": productivity,
        "transportation_cost": transportation_cost,
    }
