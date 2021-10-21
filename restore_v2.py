import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import numpy as np
from collections import defaultdict
import math
import random
import copy as cp


class Edge_Graph:
    """
        这个类用于存储电网中每一个bus之间的路径edges，
        以及线路上的阻抗 weight
    """
    def __init__(self):
        self.edges = defaultdict(list)
        self.weights = {}

    def add_edge(self, from_node, to_node, weight):
        # Note: assumes edges are bi-directional
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.weights[(from_node, to_node)] = weight
        self.weights[(to_node, from_node)] = weight


def dijsktra(graph, initial, end):
    """
    :param graph: bus节点图
    :param initial: 起点bus
    :param end: 终点bus
    :return: 从开始节点到终点的最短路径以及路径上的阻抗 path_and_weight
            path_and_weight = {'path': path, 'total_imp': total_weight}
    """
    # shortest paths is a dict of nodes
    # whose value is a tuple of (previous node, weight)
    shortest_paths = {initial: (None, 0)}
    current_node = initial
    visited = set()

    while current_node != end:
        visited.add(current_node)
        destinations = graph.edges[current_node]
        weight_to_current_node = shortest_paths[current_node][1]
        for next_node in destinations:
            weight = graph.weights[(current_node, next_node)] + weight_to_current_node
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, weight)
            else:
                current_shortest_weight = shortest_paths[next_node][1]
                if current_shortest_weight > weight:
                    shortest_paths[next_node] = (current_node, weight)
        next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
        if not next_destinations:
            return "Route Not Possible"
        # next node is the destination with the lowest weigh
        current_node = min(next_destinations, key=lambda k: next_destinations[k][1])

    # Work back through destinations in shortest path
    path = []
    total_weight = 0
    while current_node is not None:
        path.append(current_node)
        next_node = shortest_paths[current_node][0]
        total_weight = total_weight + shortest_paths[current_node][1]
        current_node = next_node
    # Reverse path
    path = path[::-1]
    # Return path as well as total weight
    path_and_weight = {'path': path, 'total_imp': total_weight}
    # print("Path&Impedence", path_and_weight)
    # return path
    return path_and_weight


class Graph_load2bus:
    """
    这个类用于生成load2bus的图结构，以及输出其路径
    """
    def __init__(self, vertices):
        # No. of vertices
        self.V = vertices

        # default dictionary to store graph
        self.graph = defaultdict(list)
        self.path_list = []

    # function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)

    '''
        A recursive function to print all paths from 'u' to 'd'.
         visited[] keeps track of vertices in current path. 
         path[] stores actual vertices and path_index is current index in path[]
    '''

    def printAllPathsUtil(self, u, d, visited, path):
        # Mark the current node as visited and store in path
        visited[u] = True
        path.append(u)
        # If current vertex is same as destination, then print
        # current path[]
        if u == d:
            # print('path', path)
            self.path_list.append(path.copy())
        else:
            # If current vertex is not destination
            # Recur for all the vertices adjacent to this vertex
            for i in self.graph[u]:
                if not visited[i]:
                    self.printAllPathsUtil(i, d, visited, path)

            # Remove current vertex from path[] and mark it as unvisited
            path.pop()
            visited[u] = False

    # Prints all paths from 's' to 'd'
    def printAllPaths(self, s, d):
        # s: gen 对应的bus, d: load 对应的bus
        self.path_list = []
        # Mark all the vertices as not visited
        visited = [False] * self.V

        # Create an array to store paths
        path = []

        # Call the recursive helper function to print all paths
        self.printAllPathsUtil(s, d, visited, path)
        return self.path_list

def creatNet(file=None):
    load_priority = None
    static_data = None
    if (file):
        # 这里是函数中使用pp部分
        net = pp.create_empty_network()
        restoration_file = pd.ExcelFile(file)

        # Creating Buses
        exc_bus = pd.read_excel(restoration_file, sheet_name='bus')
        exc_bus.sort_values(by=['bus'], inplace=True)

        for index, row in exc_bus.iterrows():
            pp.create_bus(net, vn_kv=row['vn_kv'], name=row['name'], in_service=True)

        # Creating External grid
        exc_grid = pd.read_excel(restoration_file, sheet_name='externalgrid')

        for index, row in exc_grid.iterrows():
            pp.create_ext_grid(
                net,
                bus=row['bus'],
                vm_pu=row['vm_pu'],
                va_degree=row['va_degree'],
                max_p_mw=row['max_p_mw'],
                min_p_mw=row['min_p_mw'],
                max_q_mvar=row['max_q_mvar'],
                min_q_mvar=row['min_q_mvar'],
                name=row['name'],
                in_service=True
            )
            ext_grid_bus = row['bus']

        # Creating Generators
        exc_gen = pd.read_excel(restoration_file, sheet_name='generator')
        # exc_gen = exc_gen.drop([11])    # todo 这里还有BUG，只是针对原生数据进行的处理
        for index, row in exc_gen.iterrows():
            pp.create_gen(
                net,
                bus=row['bus'],
                p_mw=row['p_mw'],
                vm_pu=row['vm_pu'],
                sn_mva=row['sn_mva'],
                name=row['name'],
                max_q_mvar=row['max_q_mvar'],
                min_q_mvar=row['min_q_mvar'],
                in_service=True
            )
        # Creating Loads
        dat = []
        exc_load = pd.read_excel(restoration_file, sheet_name='load')
        for index, row in exc_load.iterrows():
            pp.create_load(
                net,
                bus=row['bus'],
                p_mw=row['p_mw'],
                q_mvar=row['q_mvar'],
                const_z_percent=row['const_z_percent'],
                const_i_percent=row['const_i_percent'],
                sn_mva=row['sn_mva'],
                # scaling=row['scaling'],
                name=row['name'],
                in_service=True
            )
            dat.append([row['bus'], row['priority']])

        # Creating Lines
        exc_line = pd.read_excel(restoration_file, sheet_name='line')
        for index, row in exc_line.iterrows():
            pp.create_line_from_parameters(
                net,
                from_bus=row['from_bus'],
                to_bus=row['to_bus'],
                length_km=row['length_km'],
                r_ohm_per_km=row['r_ohm_per_km'],
                x_ohm_per_km=row['x_ohm_per_km'],
                c_nf_per_km=row['c_nf_per_km'],
                max_i_ka=row['max_i_ka'],
                name=row['name'],
                in_service=True
            )

        # Creating Transformers
        exc_trans = pd.read_excel(restoration_file, sheet_name='transformer')
        for index, row in exc_trans.iterrows():
            pp.create_transformer_from_parameters(
                net,
                hv_bus=row['hv_bus'],
                lv_bus=row['lv_bus'],
                name=row['name'],
                sn_mva=row['sn_mva'],
                vn_hv_kv=row['vn_hv_kv'],
                vn_lv_kv=row['vn_lv_kv'],
                vkr_percent=row['vkr_percent'],
                vk_percent=row['vk_percent'],
                pfe_kw=row['pfe_kw'],
                i0_percent=row['i0_percent'],
                shift_degree=row['shift_degree'],
                in_service=True,
                tap_side=row['tap_side']
            )
        sorted_motor = creat_motor_load(restoration_file)
        static_motor = []
        for index, row in net.load.iterrows():
            no_of_motors = len(sorted_motor[sorted_motor['load_bus'] == row['bus']])
            for i in range(0, no_of_motors):
                static_motor_row = [
                    (row['p_mw'] - sum(sorted_motor.loc[sorted_motor['load_bus'] == row['bus'], 'p_total'])) * (
                            1 / no_of_motors),
                    (row['q_mvar'] - sum(sorted_motor.loc[sorted_motor['load_bus'] == row['bus'], 'q_total'])) * (
                            1 / no_of_motors),
                    row['bus'], index, 'N']
                static_motor.append(static_motor_row)
        static_data = pd.DataFrame(static_motor, columns=['p', 'q', 'load_bus', 'id', 'processed'])
        load_priority = pd.DataFrame(dat, columns=['load_bus', 'priority'])
        load_priority['processed'] = 'N'
        load_priority = load_priority.sort_values(by=['priority'])
    else:
        net = pn.case_ieee30()

    return net, static_data, load_priority

def creat_motor_load(restoration_file):
    motors = pd.read_excel(restoration_file, sheet_name='motorload')

    motors['p'] = motors['motor_hp'] * 0.000746
    motors['q'] = motors['p'] * np.tan(np.arccos(motors['power_factor_full_load']))
    motors['p_total'] = motors['p'] * motors['no_of_motors']
    motors['q_total'] = motors['q'] * motors['no_of_motors']
    motors['p_inrush'] = motors['q'] * motors['no_of_motors'] * np.sqrt(3)
    motors['irated'] = ((motors['motor_hp'] * 746) / motors['power_factor_full_load']) / (
            np.sqrt(3) * motors['voltage_kv'] * 1000 * motors['efficiency_full_load'])
    motors['p_inrush'] = motors['voltage_kv'] * np.sqrt(3) * motors['irated'] * 6 * motors[
        'power_factor_locked_rotor'] * (1 / (np.power(10, 6))) * 1000
    motors['q_inrush'] = motors['voltage_kv'] * np.sqrt(3) * motors['irated'] * 6 * np.sin(
        np.arccos(motors['power_factor_locked_rotor'])) * (1 / np.power(10, 6)) * 1000
    motors['p_inrush_tot'] = motors['p_inrush'] * motors['no_of_motors']
    motors['q_inrush_tot'] = motors['q_inrush'] * motors['no_of_motors']
    motors['processed'] = 'N'
    motors_renamed = motors.rename(columns={'motor_hp': 'motor'})
    sorted_motor = motors_renamed.groupby(["load_bus"]).apply(
        lambda x: x.sort_values(["q_inrush_tot"], ascending=False)).reset_index(drop=True)

    return sorted_motor

def restore(file=None):
    net, static_data, load_priority = creatNet(file)
    pp.runpp(net)

