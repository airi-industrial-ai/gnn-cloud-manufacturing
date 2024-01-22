import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import openpyxl
import random

def read_fatahi_dataset(path_to_file, sheet_names = [
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
    ]):
    """
    Fatahi Valilai, Omid. “Dataset for Logistics and Manufacturing 
    Service Composition”. 17 Mar. 2021. Web. 9 June 2023.
    """
    
    res = []
    for sheet_name in tqdm(sheet_names):
        res.append(_read_sheet(path_to_file, sheet_name))
    return res


def create_distance_matrix(n_cities):
    
    cities = np.array([[random.randint(0, 2000), random.randint(0, 2000)] for i in range(n_cities)])
    
    distances = np.zeros((n_cities, n_cities), dtype=int)
    for i in range(n_cities):
        for j in range(i+1, n_cities):
            distances[j][i] = distances[i][j] = round(np.linalg.norm(cities[i] - cities[j]))
    return distances

def create_excel_table(n_operations, n_suboperations, n_cities, filepath,  n_problem=1):
    # Создаем новый файл Excel
    wb = openpyxl.load_workbook(filepath)

    # Выбираем активный лист
    sheet_name = f"{n_operations},{n_suboperations},{n_cities}-{n_problem}"
    sheet = wb.create_sheet(title=sheet_name)
    
    sheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=16)

    sheet.cell(row=1, column=1).value = f"problem No.{n_problem} with {n_operations} Operation, {n_suboperations} Sub-operations, {n_cities} Cities"
    
    start_oper = 5
    
    # Создаем симметричную матрицу расстояний
    distances = create_distance_matrix(n_cities)
    
    # Заполняем заголовки столбцов
    for i in range(1, n_operations + 1):
        sheet.cell(row=start_oper, column=i+1).value = f"Operation{i}"
    for i in range(start_oper, n_suboperations + start_oper):
        sheet.cell(row=i+1, column=1).value = f"Sub-operation{i}"
    
    n = np.random.uniform(0.2, 0.8)   
    for i in range(start_oper, n_suboperations + start_oper):
        for j in range(2, n_operations + 2):
                sheet.cell(row=i+1, column=j).value = random.choices([0,1], weights=[1-n, n])[0]

    
    sheet.merge_cells(start_row=start_oper-1, start_column=1, end_row=start_oper-1, end_column=n_operations+1)
    sheet.cell(row=start_oper-1, column=1).value = "Operations Mat"
    
    start_dist = n_suboperations + start_oper+4
    # Заполняем матрицу расстояний
    for i in range(n_cities):
        for j in range(n_cities):
            sheet.cell(row=i+start_dist+1, column=j+2).value = distances[i][j]
            
    for i in range(1, n_cities + 1):
        sheet.cell(row=start_dist, column=i+1).value = f"city{i}"
        sheet.cell(row=i+start_dist, column=1).value = f"city{i}"

    sheet.merge_cells(start_row=start_dist-1, start_column=1, end_row=start_dist-1, end_column=n_cities+1)
    sheet.cell(row=start_dist-1, column=1).value = "Distances Mat"
    
    start_times = n_cities + start_dist+ 4
    
    for i in range(1, n_cities + 1):
        sheet.cell(row=start_times, column=i+1).value = f"city{i}"
    for i in range(start_times, n_suboperations + start_times):
        sheet.cell(row=i+1, column=1).value = f"Sub-operation{i}"
    
    sheet.merge_cells(start_row=start_times-1, start_column=1, end_row=start_times-1, end_column=n_cities+1)
    sheet.cell(row=start_times-1, column=1).value = "Times Mat"
    
    p = np.random.rand() 
    costs = [[0 for j in range(n_cities)] for i in range(n_suboperations)]
    
    for i in range(start_times, n_suboperations + start_times):
        for j in range(2, n_cities + 2):
            
                if random.random() < p:
                    number = np.random.uniform(2, 10)
                    costs[i-start_times][j-2]=np.random.uniform(20, 90)
                else:
                    number = 'Inf'
                    costs[i-start_times][j-2]='Inf'
                sheet.cell(row=i+1, column=j).value = number
              
    start_cost = n_suboperations+start_times+4
    
    for i in range(n_suboperations):
        for j in range(n_cities):
            sheet.cell(row=i+start_cost+1, column=j+2).value = costs[i][j]           
    
    sheet.merge_cells(start_row=start_cost-1, start_column=1, end_row=start_cost-1, end_column=n_cities+1)
    sheet.cell(row=start_cost-1, column=1).value = "Costs Mat"
    
    for i in range(1, n_cities + 1):
        sheet.cell(row=start_cost, column=i+1).value = f"city{i}"
    for i in range(start_cost, n_suboperations + start_cost):
        sheet.cell(row=i+1, column=1).value = f"Sub-operation{i}"
    
    start_prod =  n_suboperations+start_cost+4
    
    sheet.merge_cells(start_row=start_prod-1, start_column=1, end_row=start_prod-1, end_column=n_cities)
    sheet.cell(row=start_prod-1, column=1).value = "Productivity(QoS)"
    
    for i in range(n_cities):
        sheet.cell(row=start_prod, column=i+1).value = f"city{i+1}"
        sheet.cell(row=start_prod+1, column=i+1).value = np.random.uniform(70, 100)
    
    
    
    # Сохраняем файл
    wb.save(filepath)
    return sheet_name

def solve_excel(fpath, sheet_name, city_to_op_solution, otc, shape_g):
    
    workbook = openpyxl.load_workbook(fpath)

    sheet = workbook.create_sheet(title=sheet_name)


    # Записываем данные в ячейки
    for i, (city, op) in enumerate(city_to_op_solution):
        sheet.cell(row=i+1, column=1, value=city)
        sheet.cell(row=i+1, column=2, value=op)

    # Записываем данные в ячейки
    for i, (o, t, c) in enumerate(otc):
        sheet.cell(row=i+1, column=3, value=o)
        sheet.cell(row=i+1, column=4, value=t)
        sheet.cell(row=i+1, column=5, value=c)

    sheet.cell(row=1, column=8, value=shape_g[0])
    sheet.cell(row=1, column=9, value=shape_g[1])
    sheet.cell(row=1, column=10, value=shape_g[2])

    # Сохраняем файл
    workbook.save(fpath)


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
    time_cost[np.isinf(time_cost)] = 99
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
