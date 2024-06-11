

import numpy as np
import random
random.seed(42)
np.random.seed(42)

def create_distance_matrix(n_cities):    
    cities = np.array([[random.randint(0, 2500), random.randint(0, 2500)] for i in range(n_cities)])
    distances = np.zeros((n_cities, n_cities), dtype=int)
    for i in range(n_cities):
        for j in range(i+1, n_cities):
            distances[j][i] = distances[i][j] = round(np.linalg.norm(cities[i] - cities[j]))
    return distances

def create_excel_table(wb, n_operations, n_suboperations, n_cities,  n_problem=1):
    sheet_name = f"{n_operations},{n_suboperations},{n_cities}-{n_problem}"
    sheet = wb.create_sheet(title=sheet_name)
    sheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=16)
    sheet.cell(row=1, column=1).value =\
        f"problem No.{n_problem} with {n_operations} Operation, {n_suboperations} Sub-operations, {n_cities} Cities"
    start_oper = 5
    
    # Создаем симметричную матрицу расстояний
    distances = create_distance_matrix(n_cities)
    
    # Заполняем заголовки столбцов
    for i in range(1, n_operations + 1):
        sheet.cell(row=start_oper, column=i+1).value = f"Operation{i}"
    for i in range(start_oper, n_suboperations + start_oper):
        sheet.cell(row=i+1, column=1).value = f"Sub-operation{i}"
  
    for j in range(2, n_operations + 2):
        sub_vector = generate_vector(n_suboperations)
        for i in range(start_oper, n_suboperations + start_oper):
                sheet.cell(row=i+1, column=j).value = sub_vector[i - start_oper]
    
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
    
    costs = [[0 for j in range(n_cities)] for i in range(n_suboperations)]
    
    for i in range(start_times, n_suboperations + start_times):
        cost_vec, number_vec = generate_cost_vectors(n_cities)
        for j in range(2, n_cities + 2):
            sheet.cell(row=i+1, column=j).value = number_vec[j-2]
            costs[i-start_times][j-2] = cost_vec[j-2]

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
        sheet.cell(row=start_prod+1, column=i+1).value = np.random.uniform(70, 100)/100
    return wb

def generate_cost_vectors(n):
    num_ones = random.randint(2, n-1)
    
    # Создаем вектор с num_ones единицами и (n - num_ones) нулями
    vector = np.append(np.random.uniform(20, 90, (n,)),(['Inf'] * (n - num_ones)))
    vector_2 = np.append(np.random.uniform(2, 10, (n,)),(['Inf'] * (n - num_ones)))
    
    seed = random.randint(1, 100)
    random.Random(seed).shuffle(vector)
    random.Random(seed).shuffle(vector_2)
    return vector, vector_2

def generate_vector(n):
    num_ones = random.randint(3, n-2)
    vector = [1] * num_ones + [0] * (n - num_ones)
    random.shuffle(vector)
    return vector