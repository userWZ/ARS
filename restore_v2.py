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


def creat_net(file=None, motor: bool = None):
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
                bus=row['bus'] - 1,
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
        for index, row in exc_gen.iterrows():
            pp.create_gen(
                net,
                bus=row['bus'] - 1,
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
                bus=row['bus'] - 1,
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
                from_bus=row['from_bus'] - 1,
                to_bus=row['to_bus'] - 1,
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
                hv_bus=row['hv_bus'] - 1,
                lv_bus=row['lv_bus'] - 1,
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

        # create switches between bus and line
        for frombus in net.line.from_bus:
            lineinfo = net.line.loc[net.line['from_bus'] == frombus]
            for index, lines in lineinfo.iterrows():
                duplicatecheck_frombus = net.switch[
                    (net.switch['bus'] == frombus) & (net.switch['element'] == index)]
                if duplicatecheck_frombus.empty:
                    pp.create_switch(net, bus=frombus, element=index, et='l')
                duplicatecheck_tobus = net.switch[
                    (net.switch['bus'] == lines.to_bus) & (net.switch['element'] == index)]
                if duplicatecheck_tobus.empty:
                    pp.create_switch(net, bus=lines.to_bus, element=index, et='l')
                del duplicatecheck_frombus, duplicatecheck_tobus

            # Create switches between bus and transformer
            for index, lines in net.trafo.iterrows():
                pp.create_switch(net, bus=lines.hv_bus, element=index, et='t')
                pp.create_switch(net, bus=lines.lv_bus, element=index, et='t')

        net.switch.drop_duplicates(keep=False, inplace=True)

        # creat Motor for load (if necessary)
        if motor:
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


def calculate_gen_cap(net):
    """
    计算每个发电机的cranking power, Considering few as thermal and few as gas turbine generators
    G1是黑启动电机，[G2,G3]:0.02 , 其余0.07
    :param net: 传入网络
    :return: c_pow --> df {'gen': gens, 'gen_name': gens_name, 'bus': busind, 'pow': gens_power_p, 'pow_q': gens_power_q}
    """
    gens = []
    gens_power_p = []
    gens_power_q = []
    busind = []
    gens_name = []
    for index, row in net.gen.iterrows():
        if row['name'] in ['G2', 'G3']:
            cranking_power_p = 0.02 * (abs(row['p_mw']))  #
            if math.isnan(net.gen['sn_mva'][0]):
                cranking_power_q = 0
            else:
                cranking_power_q = 0.02 * (math.sqrt(row['sn_mva'] ** 2 - row['p_mw'] ** 2))
        else:
            cranking_power_p = 0.07 * (abs(row['p_mw']))
            if math.isnan(net.gen['sn_mva'][0]):
                cranking_power_q = 0
            else:
                cranking_power_q = 0.07 * (math.sqrt(row['sn_mva'] ** 2 - row['p_mw'] ** 2))
        gens.append(index)
        gens_name.append(row['name'])
        busind.append(row['bus'])
        gens_power_p.append(cranking_power_p)
        gens_power_q.append(cranking_power_q)
    d = {'gen': gens, 'gen_name': gens_name, 'bus': busind, 'pow': gens_power_p, 'pow_q': gens_power_q}
    c_pow = pd.DataFrame(data=d)
    return c_pow


def get_path_info(net):
    """
    获取网络图结构，返回从每一个gen去每一个load的所有路径
    :param net:
    :return: graph 网络结构图
             g 节点图
             gen_load_all_paths 从gen去load的所有路径
    """
    # 计算每一个条线路的 Impedance 阻抗
    q = len(net.line)
    impedance = {}
    for a in range(0, q):
        impedance[net.line.name[a]] = abs(net.line.r_ohm_per_km[a] + 1j * net.line.x_ohm_per_km[a])
    distances = {}
    # Create Edges and Weights for the graph structure
    # Edges are buses and weights is impedance
    # 边信息第一部分，单独的线路的[from_bus, to_bus, 线路阻抗]
    edges_outer = []
    for t in net.line.from_bus:
        result = net.line.loc[net.line['from_bus'] == t]
        to_bus_data = {}
        for index, lines in result.iterrows():
            each_record = (t, lines.to_bus, impedance[lines['name']])
            edges_outer.append(each_record)
            to_bus_data[lines.to_bus] = impedance[lines['name']]
        distances[t] = to_bus_data
        del result
        del to_bus_data
    # 边信息第二部分，边上有变压器[变压器高压侧， 变压器低压侧， 0]
    for index, row in net.trafo.iterrows():
        each_record = (row['hv_bus'], row['lv_bus'], 0)
        edges_outer.append(each_record)
    # 删除重复的边信息
    edges_outer = list(set(edges_outer))
    # 创建网络边关系图
    graph = Edge_Graph()
    for edge in edges_outer:
        graph.add_edge(*edge)

    # 创建电网gen bus ---> load bus节点相互路径关系图
    g = Graph_load2bus(len(net.bus.name.unique()))
    for index, row in net.line.iterrows():
        g.addEdge(row['from_bus'], row['to_bus'])
        g.addEdge(row['to_bus'], row['from_bus'])
    for index, row in net.trafo.iterrows():
        g.addEdge(row['hv_bus'], row['lv_bus'])
        g.addEdge(row['lv_bus'], row['hv_bus'], )
    gen_load_all_paths = []
    for index, row in net.gen.iterrows():
        for index1, row1 in net.load.iterrows():
            # 得到从gen去load的所有路径
            all_paths = g.printAllPaths(row['bus'], row1['bus'])
            gen_load_path = {'gen': row['bus'], 'bus': row1['bus'], 'all_paths': all_paths}
            gen_load_all_paths.append(gen_load_path)
    return graph, g, gen_load_all_paths


def get_start_order(net, graph, c_pow, black_start=0):
    result = []
    processed = []
    gen_incl_ext_grid = []
    for each_gen in net.gen.bus:
        gen_incl_ext_grid.append(each_gen)
    gen_incl_ext_grid = list(set(gen_incl_ext_grid))
    for prev_gen in gen_incl_ext_grid:
        for curr_gen in gen_incl_ext_grid:
            if prev_gen != curr_gen and curr_gen not in processed:
                short_path = dijsktra(graph, prev_gen, curr_gen)
                # Append Source, Dest, ShortestPath, Impedence along the path, Cranking power and generator power
                shortest_path_result = dict()
                shortest_path_result['source_gen'] = prev_gen
                shortest_path_result['dest_gen'] = curr_gen
                shortest_path_result['path'] = short_path["path"]
                shortest_path_result['imp'] = short_path["total_imp"]
                shortest_path_result['c_pow'] = c_pow.loc[c_pow['bus'] == curr_gen, 'pow'].sum()
                shortest_path_result['c_pow_q'] = c_pow.loc[c_pow['bus'] == curr_gen, 'pow_q'].sum()
                if any(net.gen.bus == prev_gen):
                    shortest_path_result['source_pow'] = abs(net.gen.loc[net.gen['bus'] == prev_gen, 'p_mw'].values[0])
                else:
                    shortest_path_result['source_pow'] = abs(
                        net.res_ext_grid.loc[net.ext_grid['bus'] == prev_gen, 'p_mw'].values[0])
                if math.isnan(net.gen.loc[net.gen['bus'] == prev_gen, 'sn_mva'].values[0]):
                    shortest_path_result['source_pow_q'] = 0
                else:
                    shortest_path_result['source_pow_q'] = abs(
                        math.sqrt(net.gen.loc[net.gen['bus'] == prev_gen, 'sn_mva'].values[0] ** 2 -
                                  net.gen.loc[net.gen['bus'] == prev_gen, 'p_mw'].values[0] ** 2))
                if any(net.gen.bus == curr_gen):
                    shortest_path_result['dest_pow'] = abs(net.gen.loc[net.gen['bus'] == curr_gen, 'p_mw'].values[0])
                else:
                    shortest_path_result['dest_pow'] = abs(
                        net.res_ext_grid.loc[net.ext_grid['bus'] == curr_gen, 'p_mw'].values[0])
                if math.isnan(net.gen.loc[net.gen['bus'] == prev_gen, 'sn_mva'].values[0]):
                    shortest_path_result['source_pow_q'] = 0
                else:
                    shortest_path_result['dest_pow_q'] = abs(
                        math.sqrt(net.gen.loc[net.gen['bus'] == prev_gen, 'sn_mva'].values[0] ** 2 -
                                  net.gen.loc[net.gen['bus'] == prev_gen, 'p_mw'].values[0] ** 2))
                result.append(shortest_path_result)
            processed.append(prev_gen)

        # Opening switches based on Lowest Impedence First and lowest cranking power
        # Gen1 is the black start
        # Filtering the data based on condition that source generator is 1. Since we have to process from gen1
        conditions = {'source_gen': black_start}
        bs_result = [one_dict for one_dict in result if all(key in one_dict and conditions[key] == one_dict[key]
                                                            for key in conditions.keys())]
        # Sort the filtered data on Impedance and cranking power in ascending order
        bs_result_sorted = sorted(bs_result, key=lambda i: (i['imp']))
        start_short_path = {}
        path = [0]
        total_imp = 0
        for eachrow in bs_result_sorted:
            path.append(eachrow['dest_gen'])
            total_imp = total_imp + eachrow['imp']
        start_short_path['path'] = path
        start_short_path['total_imp'] = total_imp
        net.switch['closed'] = False
        net.gen['in_service'] = False
        abs_path = []
        for x in start_short_path['path']:
            for i, row in net.gen.iterrows():
                if row['bus'] == x:
                    if len(abs_path) == 0:
                        abs_path.append(net.ext_grid['name'].iloc[0])
                    if row['name'] != net.ext_grid['name'].iloc[0]:
                        abs_path.append(row['name'])
        return bs_result_sorted, abs_path


def start_grid(net, bs_result_sorted):
    net.gen.loc[net.gen['slack'] == True, 'in_service'] = True
    net.load.in_service = False
    # 创建空表
    available_gen = pd.DataFrame(columns=net.gen.columns.values)
    available_gen = available_gen.append(net.gen.loc[net.gen['slack'] == True])
    unprocessed_gen = cp.deepcopy(net.gen)
    unprocessed_gen['q'] = None
    for index, row in unprocessed_gen.iterrows():
        if math.isnan(row['sn_mva']):
            unprocessed_gen['q'][index] = 0
        else:
            unprocessed_gen['q'][index] = pd.eval(np.sqrt(row['sn_mva'] ** 2 - row['p_mw'] ** 2))
    cond = unprocessed_gen['name'].isin(available_gen['name'])
    unprocessed_gen = unprocessed_gen.drop(unprocessed_gen[cond].index)
    unprocessed_load = cp.deepcopy(net.load)
    not_completed_load = cp.deepcopy(net.load)
    net_copy = cp.deepcopy(net)
    iteration = 0
    rest_output = []
    rest_col_names = ['iteration', 'gen_turned_on', 'eff_gen_cap_p', 'eff_gen_cap_q', 'cranking_power_provided_gen',
                      'cranking_power_p', 'cranking_power_q', 'Load_Name', 'motor_group', 'pli_mw', 'qli_mvar',
                      'p_mw', 'q_mw', 'lp_mw', 'lq_mvar', 'pr_mw', 'qr_mvar', 'Voltage_Drop', 'Voltage_Drop_steady']

    
def restore(file=None):
    # 创建网络
    net, static_data, load_priority = creat_net(file, True)
    pp.runpp(net)
    # 计算发电机的cranking power
    cap_pow = calculate_gen_cap(net)
    # 获取从发电机去load的路径信息
    graph, g, gen_load_all_path = get_path_info(net)
    bs_result_sorted, abs_path = get_start_order(net, graph, cap_pow)
    # 开始启动电机
    start_grid(net, bs_result_sorted)


if __name__ == '__main__':
    config_file = 'IEEE30_2.xlsx'
    restore(config_file)
