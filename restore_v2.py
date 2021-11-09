import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import numpy as np
from collections import defaultdict
import math
import random
import copy as cp
import json
from json import JSONEncoder


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


def calc_vd(net, valid_path):
    net_copy = net
    line_index = net_copy.line[
        (net_copy.line['from_bus'] == valid_path[-2]) &
        (net_copy.line['to_bus'] == valid_path[-1])].index.tolist()

    if len(line_index) == 0:
        line_index = net_copy.line[
            (net_copy.line['to_bus'] == valid_path[-2]) &
            (net_copy.line['from_bus'] == valid_path[-1])].index.tolist()

        if len(line_index) == 0:
            line_index = net_copy.trafo[
                (net_copy.trafo['lv_bus'] == valid_path[-2]) &
                (net_copy.trafo['hv_bus'] == valid_path[-1])].index.tolist()

            if len(line_index) == 0:
                line_index = net_copy.trafo[
                    (net_copy.trafo['hv_bus'] == valid_path[-2]) &
                    (net_copy.trafo['lv_bus'] == valid_path[-1])].index.tolist()
                line_result = net_copy.res_trafo.iloc[line_index,]
                try:
                    vd = round(((line_result['vm_hv_pu'] - line_result[
                        'vm_lv_pu']) / line_result['vm_hv_pu']) * 100,
                               2).values[0]
                except IndexError:
                    vd = 0
            else:
                line_result = net_copy.res_line.iloc[line_index,]
                try:
                    vd = round(((line_result['vm_from_pu'] -
                                 line_result['vm_to_pu']) / line_result[
                                    'vm_to_pu']) * 100, 2).values[0]
                except IndexError:
                    vd = 0

        else:
            line_result = net_copy.res_line.iloc[line_index,]
            try:
                vd = round(((line_result['vm_from_pu'] -
                             line_result['vm_to_pu']) / line_result[
                                'vm_to_pu']) * 100, 2).values[0]
            except IndexError:
                vd = 0
    else:
        line_result = net_copy.res_line.iloc[line_index,]
        try:
            vd = round(((line_result['vm_from_pu'] -
                         line_result['vm_to_pu']) / line_result[
                            'vm_to_pu']) * 100, 2).values[0]
        except IndexError:
            vd = 0
    return vd


class BlackStartGrid:
    def __init__(self, file, has_motor=False, black_start=0):
        self.file = file
        self.has_motor = has_motor
        self.black_start = 0
        self.net, self.load_priority, self.Res_info = self.creat_net()
        self.motor_info = None if not has_motor else self.creat_motor_load()
        self.graph, self.g, self.gen_load_all_paths = self.get_path_info()
        self.static_data = self.get_static_load_info()
        self.c_pow = self.calculate_gen_cap()
        self.result = None
        self.rest_output = None
        self.open_path = []
        self.load_info = []
        self.bus_info = []
        self.gen_info = []

    class Result:
        def __init__(self, item_num, net, priority):
            self.item_num = item_num
            self.load_priority = priority
            self.index = self.create_index()
            self.bus_static = self.get_static_info(net, 'bus')
            self.gen_static = self.get_static_info(net, 'gen')
            self.line_static = self.get_static_info(net, 'line')
            self.load_static = self.get_static_info(net, 'load')
            self.bus_dynamic = []
            self.gen_dynamic = []
            self.line_dynamic = []
            self.load_dynamic = []

        def add_dynamic_data(self, net, item):
            data = net['res_' + str(item)]
            data.index = self.index[item]
            if item == 'line':
                orientation = []
                for index, row in data.iterrows():
                    if row['vm_from_pu'] and row['vm_to_pu']:
                        if row['vm_from_pu'] >= row['vm_to_pu']:
                            orientation.append(1)
                        else:
                            orientation.append(2)
                    else:
                        orientation.append(0)
                data['orientation'] = orientation
                data = data.to_dict(orient='index')
                self.line_dynamic.append(data)
                return
            data = data.to_dict(orient='index')
            if item == 'bus':
                self.bus_dynamic.append(data)
            elif item == 'gen':
                self.gen_dynamic.append(data)
            elif item == 'load':
                self.load_dynamic.append(data)

        def create_index(self):
            head = dict()
            head['bus'] = ['bus' + str(i) for i in range(0, self.item_num['bus'])]
            head['gen'] = ['gen' + str(i) for i in range(0, self.item_num['gen'])]
            head['line'] = ['line' + str(i) for i in range(0, self.item_num['line'])]
            head['load'] = ['load' + str(i) for i in range(0, self.item_num['load'])]
            return head

        def get_static_info(self, net, item):
            static = None
            if item == 'bus':
                static = net.bus[['zone']]
            elif item == 'load':
                df1 = net.load[['bus', 'p_mw', 'q_mvar']]
                df2 = self.load_priority['priority']
                static = pd.concat([df1, df2], axis=1)
                static = static.rename(columns={'p_mw': 'p_steady_mw', 'q_mvar': 'q_steady_mw'})
            elif item == 'gen':
                static = net.gen[['bus', 'max_q_mvar', 'min_q_mvar']]
                res = []
                for index, row in static.iterrows():
                    if row['bus'] == 0:
                        res.append(True)
                    else:
                        res.append(False)
                static['is_black_start'] = res
            elif item == 'line':
                static = net.line[['from_bus', 'to_bus', 'length_km', 'max_i_ka', 'r_ohm_per_km', 'x_ohm_per_km',
                                   'c_nf_per_km']]
            static.index = self.index[item]
            static = static.to_dict(orient='index')
            return static

        def save_json(self, item):
            json_file = open('result/' + item + '.json', 'w')
            if item == 'bus':
                # 先写入静态数据
                new_data = {'static': self.bus_static, 'dynamic': self.bus_dynamic}
                json_str = json.dumps(new_data)
                json_file.write(json_str)
            elif item == 'load':
                # 先写入静态数据
                new_data = {'static': self.load_static, 'dynamic': self.load_dynamic}
                json_str = json.dumps(new_data)
                json_file.write(json_str)
            elif item == 'gen':
                # 先写入静态数据
                new_data = {'static': self.gen_static, 'dynamic': self.gen_dynamic}
                json_str = json.dumps(new_data)
                json_file.write(json_str)
            elif item == 'line':
                # 先写入静态数据
                new_data = {'static': self.line_static, 'dynamic': self.line_dynamic}
                json_str = json.dumps(new_data)
                json_file.write(json_str)




    class EdgeGraph:
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

    class GraphLoad2Bus:
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

    def creat_net(self):
        load_priority = None
        item_num = dict()
        if self.file:
            # 这里是函数中使用pp部分
            net = pp.create_empty_network()
            restoration_file = pd.ExcelFile(self.file)

            # Creating Buses
            exc_bus = pd.read_excel(restoration_file, sheet_name='bus')
            exc_bus.sort_values(by=['bus'], inplace=True)

            for index, row in exc_bus.iterrows():
                pp.create_bus(net,
                              vn_kv=row['vn_kv'],
                              name=row['name'],
                              zone=row['name_of_bus'],
                              in_service=True)
            item_num['bus'] = len(exc_bus)
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
                ext_grid_bus = row['bus'] - 1

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
                    in_service=True,
                    slack=row['slack']
                )
            # Creating Loads
            item_num['gen'] = len(exc_gen)
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
                dat.append([row['bus'] - 1, row['priority']])
            item_num['load'] = len(exc_load)
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
            item_num['line'] = len(exc_line)
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

            # Create internal table for load priority
            load_priority = pd.DataFrame(dat, columns=['load_bus', 'priority'])
            load_priority['processed'] = 'N'
            load_priority = load_priority.sort_values(by=['priority'])

        else:
            net = pn.case_ieee30()

        return net, load_priority, self.Result(item_num, net, load_priority)

    def creat_motor_load(self):
        motors = pd.read_excel(self.file, sheet_name='motorload')

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
        sorted_motor['load_bus'] = sorted_motor['load_bus'] - 1
        return sorted_motor

    def get_static_load_info(self):
        """
        获取与电机相关的信息
        :return: static_data静态负载信息
        """
        static_data = None
        sorted_motor = self.motor_info
        static_info = []
        for index, row in self.net.load.iterrows():
            if self.has_motor:
                no_of_motors = len(sorted_motor[sorted_motor['load_bus'] == row['bus']])
                for i in range(0, no_of_motors):
                    static_info_row = [
                        (row['p_mw'] - sum(sorted_motor.loc[sorted_motor['load_bus'] == row['bus'], 'p_total'])) * (
                                1 / no_of_motors),
                        (row['q_mvar'] - sum(sorted_motor.loc[sorted_motor['load_bus'] == row['bus'], 'q_total'])) * (
                                1 / no_of_motors),
                        row['bus'], index, 'N']
                    static_info.append(static_info_row)
            else:
                static_info_row = [row['p_mw'], row['q_mvar'], row['bus'], index, 'N']
                static_info.append(static_info_row)

        static_data = pd.DataFrame(static_info, columns=['p', 'q', 'load_bus', 'id', 'processed'])
        return static_data

    def calculate_gen_cap(self):
        """
        计算每个发电机的cranking power, Considering few as thermal and few as gas turbine generators
        G1是黑启动电机，[G2,G3]:0.02 , 其余0.07
        :return: c_pow --> df {'gen': gens, 'gen_name': gens_name, 'bus': busind, 'pow': gens_power_p, 'pow_q': gens_power_q}
        """
        gens = []
        gens_power_p = []
        gens_power_q = []
        busind = []
        gens_name = []
        for index, row in self.net.gen.iterrows():
            if row['name'] in ['G2', 'G3']:
                cranking_power_p = 0.02 * (abs(row['p_mw']))  #
                if math.isnan(self.net.gen['sn_mva'][0]):
                    cranking_power_q = 0
                else:
                    cranking_power_q = 0.02 * (math.sqrt(row['sn_mva'] ** 2 - row['p_mw'] ** 2))
            else:
                cranking_power_p = 0.07 * (abs(row['p_mw']))
                if math.isnan(self.net.gen['sn_mva'][0]):
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

    def get_path_info(self):
        """
        获取网络图结构，返回从每一个gen去每一个load的所有路径
        :return: graph 网络结构图
                 g 节点图
                 gen_load_all_paths 从gen去load的所有路径
        """
        # 计算每一个条线路的 Impedance 阻抗
        net = self.net
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
        graph = self.EdgeGraph()
        for edge in edges_outer:
            graph.add_edge(*edge)

        # 创建电网gen bus ---> load bus节点相互路径关系图
        g = self.GraphLoad2Bus(len(net.bus.name.unique()))
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

    def get_start_order(self):
        net = self.net
        result = []
        processed = []
        gen_incl_ext_grid = []
        c_pow = self.c_pow
        for each_gen in net.gen.bus:
            gen_incl_ext_grid.append(each_gen)
        gen_incl_ext_grid = list(set(gen_incl_ext_grid))
        for prev_gen in gen_incl_ext_grid:
            for curr_gen in gen_incl_ext_grid:
                if prev_gen != curr_gen and curr_gen not in processed:
                    short_path = dijsktra(self.graph, prev_gen, curr_gen)
                    # Append Source, Dest, ShortestPath, Impedence along the path, Cranking power and generator power
                    shortest_path_result = dict()
                    shortest_path_result['source_gen'] = prev_gen
                    shortest_path_result['dest_gen'] = curr_gen
                    shortest_path_result['path'] = short_path["path"]
                    shortest_path_result['imp'] = short_path["total_imp"]
                    shortest_path_result['c_pow'] = c_pow.loc[c_pow['bus'] == curr_gen, 'pow'].sum()
                    shortest_path_result['c_pow_q'] = c_pow.loc[c_pow['bus'] == curr_gen, 'pow_q'].sum()
                    if any(net.gen.bus == prev_gen):
                        shortest_path_result['source_pow'] = abs(
                            net.gen.loc[net.gen['bus'] == prev_gen, 'p_mw'].values[0])
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
                        shortest_path_result['dest_pow'] = abs(
                            net.gen.loc[net.gen['bus'] == curr_gen, 'p_mw'].values[0])
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
            conditions = {'source_gen': self.black_start}
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
            return bs_result_sorted, abs_path, start_short_path

    def start_grid(self, bs_result_sorted, short_path):
        net = self.net
        sorted_motor = self.motor_info
        static_data = self.static_data
        net.gen.loc[net.gen['slack'] == True, 'in_service'] = True
        net.load.in_service = False
        # 创建空表
        available_gen = pd.DataFrame(columns=net.gen.columns.values)
        # available_gen = available_gen.append(net.gen.loc[net.gen['slack'] == True])
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
                          'cranking_power_p', 'cranking_power_q', 'Load_Name',
                          'p_mw', 'q_mw', 'pr_mw', 'qr_mvar', 'Voltage_Drop_steady']
        for eachrow in bs_result_sorted:
            # step4 Energize Transmission Line
            restoration_path = eachrow.get('path')
            for rest_var in range(0, len(restoration_path)):
                # 存在未开启电机
                if restoration_path[rest_var] not in list(available_gen.bus):
                    if restoration_path[rest_var] in list(unprocessed_gen.bus):
                        current_gen = restoration_path[rest_var]

                    # 判断是否路径已到达目标gen
                    if rest_var < len(restoration_path) - 1:
                        if restoration_path[rest_var + 1] in short_path['path']:
                            next_gen = restoration_path[rest_var + 1]
                    else:
                        sp = short_path['path']
                        if sp.index(current_gen) < len(sp) - 1:
                            if self.black_start == sp[sp.index(current_gen) + 1]:
                                next_gen = sp[sp.index(current_gen) + 2]
                            else:
                                next_gen = sp[sp.index(current_gen) + 1]

                    # 启动当前发电机
                    net_copy.gen.loc[net.gen['bus'] == current_gen, 'in_service'] = True
                    available_gen = available_gen.append(net_copy.gen.loc[net.gen['bus'] == current_gen], sort=True)
                    available_gen['q'] = pd.eval(np.sqrt(available_gen['sn_mva'] ** 2 - available_gen['p_mw'] ** 2))
                    # 更新未开机发电机数组
                    cond = unprocessed_gen['name'].isin(available_gen['name'])
                    if cond.any():
                        self.gen_info.append({
                            'iteration': iteration,
                            'gen': self.bus_to_name('gen', current_gen),
                            'open_path': [node + 1 for node in eachrow.get('path')],
                            'imp': eachrow['imp']
                        })
                    unprocessed_gen = unprocessed_gen.drop(unprocessed_gen[cond].index)

                    # Calculate Available generation capacity, processed load and effective generation capability
                    gen_capacity = abs(available_gen['p_mw'].sum())
                    gen_capacity_q = abs(available_gen['q'].sum())
                    c_pow = self.c_pow
                    cranking_power = abs(c_pow.loc[c_pow['bus'] == next_gen, 'pow'].sum())
                    cranking_power_q = abs(c_pow.loc[c_pow['bus'] == next_gen, 'pow_q'].sum())

                    try:
                        processed_load_steadystate_p = static_data.query("processed == 'Y'")['p'].sum()
                        processed_load_steadystate_q = static_data.query("processed == 'Y'")['q'].sum()
                    except IndexError:
                        processed_load_steadystate_p = 0
                        processed_load_steadystate_q = 0

                    if self.has_motor:
                        try:
                            processed_load_steadystate_mot_p = sorted_motor.query("processed == 'Y'")['p_total'].sum()
                            processed_load_steadystate_mot_q = sorted_motor.query("processed == 'Y'")['q_total'].sum()
                        except IndexError:
                            processed_load_steadystate_mot_p = 0
                            processed_load_steadystate_mot_q = 0
                        eff_gen_cap = gen_capacity - cranking_power - processed_load_steadystate_p - processed_load_steadystate_mot_p
                        eff_gen_cap_q = gen_capacity_q - cranking_power_q - processed_load_steadystate_q - processed_load_steadystate_mot_q
                    else:
                        eff_gen_cap = gen_capacity - cranking_power - processed_load_steadystate_p
                        eff_gen_cap_q = gen_capacity_q - cranking_power_q - processed_load_steadystate_q
                    load_processed = False
                    current_load_completed = False
                    insufficient_capacity = False
                    if len(unprocessed_gen) == 0:
                        print('发电机全部开启', len(unprocessed_gen))
                    if len(unprocessed_load) == 0:
                        print('负载全部开启', len(unprocessed_load))

                    # step 8&9  Select the loads to pick up
                    for l_index, l_row in self.load_priority.iterrows():
                        if not insufficient_capacity:
                            current_load = l_row['load_bus']
                        else:
                            break
                        random_multi = round(random.uniform(0.05, 0.1), 2)
                        if not unprocessed_load[unprocessed_load['bus'] == current_load].any().any():
                            # print('当前负载已经开启了')
                            if len(unprocessed_gen) == 0:
                                print('发电机全部打开')
                                pp.runpp(net_copy)
                                if net_copy.converged:
                                    # 写入一条潮流计算后的数据
                                    self.Res_info.add_dynamic_data(net_copy, 'bus')
                                    self.Res_info.add_dynamic_data(net_copy, 'load')
                                    self.Res_info.add_dynamic_data(net_copy, 'gen')
                                    self.Res_info.add_dynamic_data(net_copy, 'line')
                                    # 开启新电机
                                    iteration = iteration + 1
                                    normalvd = calc_vd(net_copy, short_path['path'])
                                    rest_row = [[
                                        iteration,
                                        str(net_copy.gen.loc[(net_copy.gen.in_service == True),
                                                             'name'].tolist()).strip('[]'),
                                        round(eff_gen_cap, 2),
                                        round(eff_gen_cap_q, 2),
                                        '-',
                                        round(cranking_power, 2),
                                        round(cranking_power_q, 2),
                                        net_copy.load.loc[(net_copy.load.bus == int(current_load)),
                                                          'name'].values[0],
                                        round(picked_steady_load1, 2),
                                        round(picked_steady_load1_q, 2),
                                        round(picked_steady_load1 + (
                                                static_p * random_multi), 2),
                                        round(picked_steady_load1_q + (
                                                static_q * random_multi), 2),
                                        normalvd
                                    ]]
                                    rest_df = pd.DataFrame(rest_row, columns=rest_col_names)
                                    try:
                                        rest_output = rest_output.append(rest_df, ignore_index=False)
                                    except:
                                        rest_output = rest_df.copy()
                                break
                            else:
                                continue
                        for eachload_paths in self.gen_load_all_paths:
                            # 从gen 去 load

                            if ((available_gen[available_gen['bus'] == eachload_paths.get('gen')].any().any())
                                    & (eachload_paths.get('bus') == current_load)):
                                # 找到 从current_gen去current_load的path
                                all_paths_arr = eachload_paths.get('all_paths')
                                valid_path = []
                                unprocessed_gen_set = set(unprocessed_gen['bus'])
                                unprocessed_load_set = set(unprocessed_load['bus'])
                                trans_hv = set(net_copy.trafo['hv_bus'])
                                trans_lv = set(net_copy.trafo['lv_bus'])
                                for i in range(0, len(all_paths_arr)):
                                    valid_path_flag = True
                                    single_path = all_paths_arr[i]
                                    single_path_set = set(single_path)
                                    single_path_set_load = set(single_path[:-1])
                                    # 检查是否path中存在未开机的gen
                                    if unprocessed_gen_set.intersection(single_path_set) \
                                            or unprocessed_load_set.intersection(single_path_set_load) \
                                            or trans_hv.intersection(single_path_set) \
                                            or trans_lv.intersection(single_path_set):
                                        valid_path_flag = False
                                        # print('路径中存在未启动的电机或负载')
                                    if valid_path_flag:
                                        valid_path = all_paths_arr[i]

                                        # print('{i}: 从{gen}启动load: {load}的有效路径path: {path}'.format(i=iteration,
                                        #                                                           gen=current_gen,
                                        #                                                           load=current_load,
                                        #                                                           path=valid_path))
                                        break
                                if not valid_path_flag:
                                    # 路径中存在未开机的电机或者负载时
                                    # 这里又判断了一遍不计算变压器的情况，不太理解
                                    for i in range(0, len(all_paths_arr)):
                                        valid_path_flag = True
                                        single_path = all_paths_arr[i]
                                        single_path_set = set(single_path)
                                        single_path_set_load = set(single_path[:-1])
                                        if unprocessed_gen_set.intersection(single_path_set) \
                                                or unprocessed_load_set.intersection(single_path_set_load):
                                            valid_path_flag = False
                                        else:
                                            for j in range(0, len(single_path) - 1):
                                                if (net_copy.trafo.loc[
                                                    (net_copy.trafo.hv_bus == single_path[j]) &
                                                    (net_copy.trafo.lv_bus == single_path[j + 1]) &
                                                    (net_copy.trafo.tap_side == 'hv')].any().any()
                                                        or net_copy.trafo.loc[
                                                            (net_copy.trafo.hv_bus == single_path[j + 1]) &
                                                            (net_copy.trafo.lv_bus == single_path[j]) &
                                                            (net_copy.trafo.tap_side == 'lv')].any().any()):
                                                    valid_path_flag = False
                                                    break
                                        if valid_path_flag:
                                            valid_path = all_paths_arr[i]
                                            # print('{i}: 从{gen}启动load: {load}的有效路径path: {path}'.format(i=iteration,
                                            #                                                           gen=current_gen,
                                            #                                                           load=current_load,
                                            #                                                           path=valid_path))
                                            break

                                if not valid_path_flag:
                                    # 找不到合理的路径， 开始寻找下一个gen_load_path
                                    continue
                                else:
                                    # 有合理路径
                                    for i in range(0, len(valid_path) - 1):
                                        if (valid_path[i] is not None) & (valid_path[i + 1] is not None):
                                            # 还没走到终点
                                            # 路径bus之间的线路
                                            line_bw_buses = net_copy.line.loc[(net_copy.line['from_bus'] ==
                                                                               valid_path[i]) & (
                                                                                      net_copy.line['to_bus'] ==
                                                                                      valid_path[i + 1])]
                                            if len(line_bw_buses) == 0:
                                                line_bw_buses = net_copy.line.loc[
                                                    (net_copy.line['from_bus'] == valid_path[i + 1]) & (
                                                            net_copy.line['to_bus'] == valid_path[i])]
                                            # 开启line的开关
                                            if len(line_bw_buses) > 0:
                                                net_copy.switch.loc[
                                                    (net_copy.switch['element'] == line_bw_buses.index[0]) &
                                                    (net_copy.switch['et'] == 'l'), 'closed'] = True
                                            # 检查路径bus之间是否有变压器
                                            trafo_bw_buses = net_copy.trafo.loc[(net_copy.trafo['hv_bus'] ==
                                                                                 valid_path[i]) &
                                                                                (net_copy.trafo['lv_bus'] == valid_path[
                                                                                    i + 1])]
                                            if len(trafo_bw_buses) == 0:
                                                trafo_bw_buses = net_copy.trafo.loc[
                                                    (net_copy.trafo['lv_bus'] == valid_path[i]) &
                                                    (net_copy.trafo['hv_bus'] == valid_path[i + 1])]
                                            # 打开变压器
                                            if len(trafo_bw_buses) > 0:
                                                net_copy.switch.loc[
                                                    (net_copy.switch['element'] == int(trafo_bw_buses.index[0])) &
                                                    (net_copy.switch['et'] == 't'), 'closed'] = True
                                            # 变压器高压侧
                                            temp_trafo_switch = net_copy.trafo.loc[
                                                net_copy.trafo.hv_bus == valid_path[i]]

                                            if len(temp_trafo_switch) == 0:
                                                temp_trafo_switch = net_copy.trafo.loc[
                                                    net_copy.trafo.lv_bus == valid_path[i]]
                                            if len(temp_trafo_switch) > 0:
                                                for ts_iter, ts_row in temp_trafo_switch.iterrows():
                                                    if ts_row['hv_bus'] == valid_path[i] and \
                                                            net_copy.load.loc[
                                                                (net_copy.load.bus == ts_row['lv_bus']) & (
                                                                        net_copy.load.in_service == True)].any().any():
                                                        trans_index = net_copy.trafo.loc[
                                                            (net_copy.trafo.hv_bus == ts_row['hv_bus']) &
                                                            (net_copy.trafo.lv_bus == ts_row['lv_bus'])].index.values
                                                        net_copy.switch.loc[
                                                            (net_copy.switch['element'] == trans_index[0]) & (
                                                                    net_copy.switch['et'] == 't'), 'closed'] = True
                                                    elif (ts_row['lv_bus'] == valid_path[i]) and (net_copy.load.loc[
                                                        (net_copy.load.bus == ts_row['hv_bus']) & (
                                                                net_copy.load.in_service ==
                                                                True)].any().any()):
                                                        trans_index = net_copy.trafo.loc[
                                                            (net_copy.trafo.hv_bus == ts_row['hv_bus']) & (
                                                                    net_copy.trafo.lv_bus == ts_row[
                                                                'lv_bus'])].index.values
                                                        net_copy.switch.loc[
                                                            (net_copy.switch['element'] == trans_index[0]) & (
                                                                    net_copy.switch['et'] == 't'), 'closed'] = True
                                    unprocessed_load.drop(
                                        unprocessed_load[unprocessed_load['bus'] == int(current_load)].index,
                                        inplace=True)
                                    # print('success! gen: {gen}, load: {load}: path: {path}'.format(gen=current_gen,
                                    #                                                                load=current_load,
                                    #                                                                path=valid_path))
                                    try:
                                        static = static_data[(static_data.load_bus == int(current_load)) & (
                                                static_data.processed == 'N')].iloc[0]
                                    except IndexError:
                                        static = None
                                    except TypeError:
                                        static = None

                                    static_p = static['p'] if static is not None else 0
                                    static_q = static['q'] if static is not None else 0
                                    if static is not None:
                                        # 开启负载
                                        net_copy.load.loc[
                                            (net_copy.load['bus'] == current_load), 'in_service'] = True
                                        picked_steady_load1 = static_p
                                        picked_steady_load1_q = static_q
                                        pp.runpp(net_copy)
                                        # 如果开启这个负载后能够潮流收敛， 写入开启时的数据
                                        if net_copy.converged:
                                            # 写入一条潮流计算后的数据
                                            self.Res_info.add_dynamic_data(net_copy, 'bus')
                                            self.Res_info.add_dynamic_data(net_copy, 'load')
                                            self.Res_info.add_dynamic_data(net_copy, 'gen')
                                            self.Res_info.add_dynamic_data(net_copy, 'line')
                                            iteration = iteration + 1
                                            rest_row = None
                                            rest_df = None
                                            normalvd = calc_vd(net_copy, valid_path)
                                            if len(rest_output) == 0:
                                                # 写入第一条数据
                                                rest_row = [[
                                                    iteration,
                                                    str(net_copy.gen.loc[(net_copy.gen.in_service == True),
                                                                         'name'].tolist()).strip('[]'),
                                                    round(eff_gen_cap, 2),
                                                    round(eff_gen_cap_q, 2),
                                                    '-' if next_gen is None else
                                                    str(c_pow.loc[c_pow['bus'] == next_gen,
                                                                  'gen_name'].tolist()).strip('[]'),
                                                    round(cranking_power, 2),
                                                    round(cranking_power_q, 2),
                                                    net_copy.load.loc[(net_copy.load.bus == int(current_load)),
                                                                      'name'].values[0],
                                                    round(picked_steady_load1, 2),
                                                    round(picked_steady_load1_q, 2),
                                                    round(picked_steady_load1 + (
                                                            static_p * random_multi), 2),
                                                    round(picked_steady_load1_q + (
                                                            static_q * random_multi), 2),
                                                    normalvd
                                                ]]
                                                rest_df = pd.DataFrame(rest_row, columns=rest_col_names)
                                            else:
                                                if rest_output.loc[rest_output['cranking_power_provided_gen'] ==
                                                                   str(c_pow.loc[c_pow[
                                                                                     'bus'] == next_gen, 'gen_name'].tolist()).strip(
                                                                       '[]')].any().any():
                                                    # 如果没有新开的gen
                                                    rest_row = [[
                                                        iteration,
                                                        '-',
                                                        round(eff_gen_cap, 2),
                                                        round(eff_gen_cap_q, 2),
                                                        '-',
                                                        '-',
                                                        '-',
                                                        net_copy.load.loc[(net_copy.load.bus == int(
                                                            current_load)), 'name'].values[0],
                                                        round(picked_steady_load1, 2),
                                                        round(picked_steady_load1_q, 2),
                                                        round(picked_steady_load1 + (
                                                                static_p * random_multi), 2),
                                                        round(picked_steady_load1_q + (
                                                                static_q * random_multi), 2),
                                                        normalvd
                                                    ]]
                                                    rest_df = pd.DataFrame(rest_row, columns=rest_col_names)
                                                else:
                                                    # 开启新电机
                                                    rest_row = [[
                                                        iteration,
                                                        str(net_copy.gen.loc[(net_copy.gen.in_service == True),
                                                                             'name'].tolist()).strip('[]'),
                                                        round(eff_gen_cap, 2),
                                                        round(eff_gen_cap_q, 2),
                                                        '-' if next_gen is None else
                                                        str(c_pow.loc[c_pow['bus'] == next_gen,
                                                                      'gen_name'].tolist()).strip('[]'),
                                                        round(cranking_power, 2),
                                                        round(cranking_power_q, 2),
                                                        net_copy.load.loc[(net_copy.load.bus == int(current_load)),
                                                                          'name'].values[0],
                                                        round(picked_steady_load1, 2),
                                                        round(picked_steady_load1_q, 2),
                                                        round(picked_steady_load1 + (
                                                                static_p * random_multi), 2),
                                                        round(picked_steady_load1_q + (
                                                                static_q * random_multi), 2),
                                                        normalvd
                                                    ]]
                                                    rest_df = pd.DataFrame(rest_row, columns=rest_col_names)
                                            try:
                                                rest_output = rest_output.append(rest_df, ignore_index=False)
                                            except:
                                                rest_output = rest_df.copy()

                                        static_data.loc[(static_data['load_bus'] == int(current_load)) &
                                                        (static_data['id'] == static['id']), 'processed'] = 'Y'
                                        static_data.loc[(static_data['load_bus'] == int(current_load)) &
                                                        (static_data['id'] == static['id']), 'p'] = static_p + (
                                                static_p * random_multi)
                                        static_data.loc[(static_data['load_bus'] == int(current_load)) &
                                                        (static_data['id'] == static['id']), 'q'] = static_q + (
                                                static_q * random_multi)
                                        self.open_path.append({
                                            'iteration': iteration,
                                            'from_gen': self.bus_to_name('gen', eachload_paths.get('gen')),
                                            'open_load': self.bus_to_name('load', current_load),
                                            'open_path': [node + 1 for node in valid_path],
                                            'imp': self.get_path_imp(valid_path)
                                        })

                                        cranking_power = abs(c_pow.loc[c_pow['bus'] == next_gen, 'pow'].sum())
                                        cranking_power_q = abs(c_pow.loc[c_pow['bus'] == next_gen, 'pow_q'].sum())
                                    else:
                                        # print('这条路径，没有未开启的负载了')
                                        break
                if len(unprocessed_load) == 0 and len(unprocessed_gen) is None:
                    print('任务完成')
                    break
        return rest_output

    def bus_to_name(self, opt, bus):
        net = self.net
        name = str(net[opt].loc[net[opt]['bus'] == bus, 'name'].tolist()).strip('[]')
        return name

    def restore(self):
        pp.runpp(self.net)
        # 获取从发电机去load的路径信息
        bs_result_sorted, abs_path, short_path = self.get_start_order()
        # 开始启动电机
        self.result = self.start_grid(bs_result_sorted, short_path)
        # print(self.open_path)
        self.Res_info.save_json('bus')
        self.Res_info.save_json('load')
        self.Res_info.save_json('line')
        self.Res_info.save_json('gen')
        return self.result, abs_path

    def get_path_imp(self, path):
        graph = self.graph
        imp = 0
        for i in range(0, len(path) - 1):
            cur_node = path[i]
            next_node = path[i + 1]
            imp = imp + self.graph.weights[(cur_node, next_node)]
        return imp


if __name__ == '__main__':
    config_file = 'IEEE30_2.xlsx'
    Black_Start = BlackStartGrid(config_file)
    Black_Start.restore()
