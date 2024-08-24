import networkx as nx
from gnncloudmanufacturing.utils import graph_from_problem
import numpy as np
import matplotlib.pyplot as plt

city_pos = np.array([
    [  115.99973719,  -341.1306974 ],
    [-1413.91669519,  -914.54016857],
    [ -103.82198498,   229.7099274 ],
    [  205.71947546,  -341.56030095],
    [ 1233.89988871,  -509.88239254],
    [ -438.31897102,   766.92359244],
    [  396.95595823,   655.93620715],
    [  161.04686815,  -106.13929762],
    [  874.58709773,   169.86873108],
    [-2044.40822386,   322.25568895],
    [  555.91129348,  -540.14241486],
    [-1273.79138966,   383.06905193],
    [ 1438.6629408 ,  -340.85072123],
    [  623.43276747,    -4.4324665 ],
    [  358.29036491,    65.6282183 ],
    [ -662.84108771,   225.00265611],
    [  420.18892386,  -372.14195066],
    [-1293.98449614,  1022.32488935],
    [  -75.06145596,   232.18427254],
    [  862.07312503,  -657.26383814]
])


def plot_problem(problem, task=None, figsize=(10,8)):
    subtask_names, city_names, graph, pos, o2o, c2o = _prepare_plot(problem, task)

    plt.figure(figsize=figsize)
    _draw_graph(subtask_names, city_names, graph, pos, o2o)
    nx.draw_networkx_edges(
        graph,
        pos=pos,
        edgelist=c2o,
        edge_color='tab:pink',
        arrows=False,
        alpha=0.1,
    )
    plt.axis('equal')
    plt.axis('off')
    plt.show()


def plot_solution(problem, gamma, task=None, figsize=(10,8)):
    subtask_names, city_names, graph, pos, o2o, c2o = _prepare_plot(problem, task)

    c2o = []
    for o, t, c in zip(*np.where(gamma == 1)):
        c2o.append((f'{c}', f'{t}_{o}'))
    
    if task is not None:
        c2o = [(u, v) for u, v in c2o if v in subtask_names]

    plt.figure(figsize=figsize)
    _draw_graph(subtask_names, city_names, graph, pos, o2o)
    nx.draw_networkx_edges(
        graph,
        pos=pos,
        edgelist=c2o,
        edge_color='tab:pink',
        arrows=False,
    )
    plt.axis('equal')
    plt.axis('off')
    plt.show()


def _draw_graph(subtask_names, city_names, graph, pos, o2o):
    nx.draw_networkx_nodes(
        graph,
        pos=pos,
        nodelist=subtask_names,
        node_size=200,
        node_color='tab:purple',
    )
    nx.draw_networkx_nodes(
        graph,
        pos=pos,
        nodelist=city_names,
        node_size=200,
        node_color='tab:orange',
    )
    nx.draw_networkx_labels(
        graph, 
        pos,
        labels={node: node for node in subtask_names},
        font_color='white', 
        font_size=7,
    )
    nx.draw_networkx_labels(
        graph, 
        pos,
        labels={node: node for node in city_names},
        font_color='white', 
        font_size=7,
    )
    nx.draw_networkx_edges(
        graph,
        pos=pos,
        edgelist=o2o,
        edge_color='tab:purple',
    )


def _prepare_plot(problem, task=None):
    dglgraph = graph_from_problem(problem)
    n_tasks = problem['n_tasks']
    n_cities = problem['n_cities']
    operation_index = dglgraph.ndata['operation_index']['o'].numpy()
    subtask_names = [f'{i}_{j}' for (i, j) in operation_index]
        
    city_names = np.arange(n_cities).astype(str).tolist()
    graph = nx.DiGraph(dglgraph.to_networkx())
    mapping = {node: name for node, name in zip(graph.nodes, subtask_names + city_names)}
    graph = nx.relabel_nodes(graph, mapping)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    
    pos = dict()
    for i, name in enumerate(subtask_names):
        pos[name] = list(operation_index[i][::-1])
    for i in range(n_cities):
        pos[str(i)] = city_pos[i]*0.003 + [5, n_tasks + 4]
    
    o2o = []
    c2o = []
    for u, v in graph.edges:
        utype, vtype = graph.nodes[u]['ntype'], graph.nodes[v]['ntype']
        if utype == 'o' and vtype == 'o':
            o2o.append((u, v))
        if utype == 's' and vtype == 'o':
            c2o.append((u, v))
    o2o = o2o[::2]
    
    if task is not None:
        subtask_names = np.array(subtask_names)[operation_index[:, 0] == task]
        graph.remove_nodes_from(
            [node for node in graph.nodes if (graph.nodes[node]['ntype'] == 'o' and node not in subtask_names)]
        )
        o2o = [(u, v) for u, v in o2o if u in subtask_names]
        c2o = [(u, v) for u, v in c2o if v in subtask_names]
    
    return subtask_names,city_names,graph,pos,o2o,c2o
