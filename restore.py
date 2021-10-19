import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import numpy as np
from collections import defaultdict
import math
import random
import copy as cp


def restore(file):
    # Graph class containing edge and weight (impedance) and algorithm to identify
    # #shortest path between edges based on Impedance
    class Graph:
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

    # Python program to print all paths from a source to destination.
    # This class represents a directed graph
    # using adjacency list representation

    class Graph1:
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
                print('path', path)
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
    if(file):
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
        pp.runpp(net)

        # Create internal table for motors
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
        static_motor = []
        static_motor_row = []
        sorted_motor = motors_renamed.groupby(["load_bus"]).apply(
            lambda x: x.sort_values(["q_inrush_tot"], ascending=False)).reset_index(drop=True)
        for index, row in net.load.iterrows():
            no_of_motors = len(sorted_motor[sorted_motor['load_bus'] == row['bus']])
            for i in range(0, no_of_motors):
                static_motor_row = [
                    (row['p_mw'] - sum(sorted_motor.loc[sorted_motor['load_bus'] == row['bus'], 'p_total'])) * (1 / no_of_motors),
                    (row['q_mvar'] - sum(sorted_motor.loc[sorted_motor['load_bus'] == row['bus'], 'q_total'])) * ( 1 / no_of_motors),
                     row['bus'], index,'N']
                static_motor.append(static_motor_row)

        static_data = pd.DataFrame(static_motor, columns=['p', 'q', 'load_bus', 'id', 'processed'])

    else:
        net = pn.case_ieee30()
        # todo 添加motorload表，添加load优先级row
    # Create internal table for load priority
    load_priority = pd.DataFrame(dat, columns=['load_bus', 'priority'])
    load_priority['processed'] = 'N'
    load_priority = load_priority.sort_values(by=['priority'])

    # Create Table with Cranking Power
    gens = []
    gens_power_p = []
    gens_power_q = []
    busind = []
    gens_name = []

    # Considering few as thermal and few as gas turbine generators
    for index, row in net.gen.iterrows():
        if row['name'] in ['G2', 'G3']:
            cranking_power_p = 0.02 * (abs(row['p_mw'])) #
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
    q = len(net.line)

    # create switches between bus and line
    for frombus in net.line.from_bus:
        lineinfo = net.line.loc[net.line['from_bus'] == frombus]
        for index, lines in lineinfo.iterrows():
            duplicatecheck_frombus = net.switch[(net.switch['bus'] == frombus) & (net.switch['element'] == index)]
            if duplicatecheck_frombus.empty:
                pp.create_switch(net, bus=frombus, element=index, et='l')
            duplicatecheck_tobus = net.switch[(net.switch['bus'] == lines.to_bus) & (net.switch['element'] == index)]
            if duplicatecheck_tobus.empty:
                pp.create_switch(net, bus=lines.to_bus, element=index, et='l')
            del duplicatecheck_frombus, duplicatecheck_tobus

    # Create switches between bus and transformer
    for index, lines in net.trafo.iterrows():
        load_index = net.load.loc[net.load['bus'] == frombus]
        pp.create_switch(net, bus=lines.hv_bus, element=index, et='t')
        pp.create_switch(net, bus=lines.lv_bus, element=index, et='t')

    net.switch.drop_duplicates(keep=False, inplace=True)

    # Create Impedance 阻抗
    q = len(net.line)
    impedance = {}
    for a in range(0, q):
        impedance[net.line.name[a]] = abs(net.line.r_ohm_per_km[a] + 1j * net.line.x_ohm_per_km[a])
    l = []
    for s in net.bus.name:
        l.append(s)
    nodes = l
    distances = {}
    to_bus_data = {}
    from_bus_data = {}

    '''
    前面是电网相关的，后面是与路径规划相关的代码
    '''
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
    graph = Graph()
    # 删除重复的边信息
    edges_outer = list(set(edges_outer))
    for edge in edges_outer:
        graph.add_edge(*edge)
    # Create a graph given in the above diagram
    no_of_buses = len(net.bus.name.unique())
    # 这个类用于点信息
    g = Graph1(no_of_buses)
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

    # Between each pair of generators perform Dijsktra to find out shortest path
    prev_gen = None
    result = []
    processed = []
    load_temp = net.load
    # Creating temp variable for net.gen.bus to include external grid bus in the list
    gen_incl_ext_grid = []
    # for each_ext_grid in net.ext_grid.bus:
    #   gen_incl_ext_grid.append(each_ext_grid)
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
                    if math.isnan(net.gen.loc[net.gen['bus'] == prev_gen, 'sn_mva'].values[0]):
                        shortest_path_result['source_pow_q'] = 0
                    else:
                        shortest_path_result['source_pow_q'] = abs(
                            math.sqrt(net.gen.loc[net.gen['bus'] == prev_gen, 'sn_mva'].values[0] ** 2
                                      - net.gen.loc[net.gen['bus'] == prev_gen, 'p_mw'].values[0] ** 2)) - math.sqrt(
                            row['sn_mva'] ** 2 - row['p_mw'] ** 2)
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
                    if math.isnan(net.gen.loc[net.gen['bus'] == prev_gen, 'sn_mva'].values[0]):
                        shortest_path_result['source_pow_q'] = 0
                    else:
                        shortest_path_result['dest_pow_q'] = abs(
                            math.sqrt(net.gen.loc[net.gen['bus'] == prev_gen, 'sn_mva'].values[0] ** 2 -
                                      net.gen.loc[net.gen['bus'] == prev_gen, 'p_mw'].values[0] ** 2))
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
                print(shortest_path_result)
        processed.append(prev_gen)

    # Opening switches based on Lowest Impedence First and lowest cranking power
    # Gen1 is the black start
    black_start = 0
    # Filtering the data based on condition that source generator is 1. Since we have to process from gen1
    conditions = {'source_gen': 1}
    bs_result = [one_dict for one_dict in result if all(key in one_dict and conditions[key] == one_dict[key]
                                                        for key in conditions.keys())]
    # Sort the filtered data on Impedance and cranking power in ascending order
    # bs_result_sorted = sorted(bs_result, key = lambda i: (i['imp'],i['c_pow']))
    bs_result_sorted = sorted(bs_result, key=lambda i: (i['imp']))
    path = []
    path.append(1)
    total_imp = 0
    for eachrow in bs_result_sorted:
        path.append(eachrow['dest_gen'])
        total_imp = total_imp + eachrow['imp']
    short_path['path'] = path
    short_path['total_imp'] = total_imp
    net.switch['closed'] = False
    net.gen['in_service'] = False

    ####### Looping Swith Logic to be added ##########
    # Loop on filtered data where source generator is 0
    # for eachrow in bs_result_sorted:
    # Get the shortest path between two generators
    net.gen.loc[net.gen['bus'] == ext_grid_bus, 'in_service'] = True
    net.gen.loc[net.gen['bus'] == ext_grid_bus, 'slack'] = True
    slack_vm_pu = net.gen.loc[net.gen['slack'] == True, 'vm_pu']
    net.load.in_service = False
    processed_bus = []
    inrush_bus = []
    load_temp = net.load.copy()
    indexnames = None
    picked_load1 = 0
    picked_load2 = 0
    available_gen = pd.DataFrame(columns=net.gen.columns.values)
    unprocessed_temp = []
    avail_list_gen = []

    unprocessed_gen = cp.deepcopy(net.gen)
    for index, row in unprocessed_gen.iterrows():
        if math.isnan(row['sn_mva']):
            row['q'] = 0
        else:
            row['q'] = pd.eval(np.sqrt(row['sn_mva'] ** 2 - row['p_mw'] ** 2))

    cond = unprocessed_gen['name'].isin(available_gen['name'])
    unprocessed_gen = unprocessed_gen.drop(unprocessed_gen[cond].index)
    unprocessed_load = cp.deepcopy(net.load)
    not_completed_load = cp.deepcopy(net.load)
    net_copy = pp.create_empty_network()
    net_copy = cp.deepcopy(net)
    iteration = float(0)
    rest_output = []
    current_load_processed = False
    current_gen = None
    next_gen = None
    rest_col_names = ['iteration', 'gen_turned_on', 'eff_gen_cap_p', 'eff_gen_cap_q', 'cranking_power_provided_gen',
                      'cranking_power_p', 'cranking_power_q', 'Load_Name', 'motor_gr oup', 'pli_mw', 'qli_mvar',
                      'p_mw',
                      'q_mw', 'lp_mw', 'lq_mvar', 'pr_mw', 'qr_mvar', 'Voltage_Drop', 'Voltage_Drop_steady']
    for eachrow in bs_result_sorted:
        restoration_path = eachrow.get('path')
        for rest_var in range(0, len(restoration_path)):
            if restoration_path[rest_var] not in list(available_gen.bus):
                current_gen = restoration_path[rest_var]
                if rest_var < len(restoration_path) - 1:
                    next_gen = restoration_path[rest_var + 1]
                else:
                    next_gen = None

                if next_gen is None:
                    sp = short_path['path']
                    if sp.index(current_gen) < len(sp) - 1:
                        if black_start == sp[sp.index(current_gen) + 1]:
                            next_gen = sp[sp.index(current_gen) + 2]
                        else:
                            next_gen = sp[sp.index(current_gen) + 1]
                net_copy.gen.loc[net.gen['bus'] == current_gen, 'in_service'] = True
                available_gen = available_gen.append(net_copy.gen.loc[net.gen['bus'] == current_gen], sort=True)
                available_gen['q'] = pd.eval(np.sqrt(available_gen['sn_mva'] ** 2 - available_gen['p_mw'] ** 2))
                cond = unprocessed_gen['name'].isin(available_gen['name'])
                unprocessed_gen = unprocessed_gen.drop(unprocessed_gen[cond].index)
                gen_capacity = float(0)
                gen_capacity_q = float(0)
                cranking_power = None
                cranking_power_q = None

                # Calculate Available generation capacity, processed load and effective generation capability
                gen_capacity = gen_capacity + abs(available_gen['p_mw'].sum())
                gen_capacity_q = gen_capacity_q + abs(available_gen['q'].sum())
                cranking_power = abs(c_pow.loc[c_pow['bus'] == next_gen, 'pow'].sum())
                cranking_power_q = abs(c_pow.loc[c_pow['bus'] == next_gen, 'pow_q'].sum())

                try:
                    processed_load_steadystate_p = static_data.query("processed == 'Y'")['p'].sum()
                    processed_load_steadystate_q = static_data.query("processed == 'Y'")['q'].sum()
                except IndexError:
                    processed_load_steadystate_p = 0
                    processed_load_steadystate_q = 0
                try:
                    processed_load_steadystate_mot_p = sorted_motor.query("processed == 'Y'")['p_total'].sum()
                    processed_load_steadystate_mot_q = sorted_motor.query("processed == 'Y'")['q_total'].sum()
                except IndexError:
                    processed_load_steadystate_mot_p = 0
                    processed_load_steadystate_mot_q = 0

                eff_gen_cap = gen_capacity - cranking_power - processed_load_steadystate_p - processed_load_steadystate_mot_p
                eff_gen_cap_q = gen_capacity_q - cranking_power_q - processed_load_steadystate_q - processed_load_steadystate_mot_q
                load_processed = False
                current_load_completed = False
                insufficient_capacity = False
                for l_index, l_row in load_priority.iterrows():
                    if insufficient_capacity == False:
                        current_load = l_row['load_bus']
                    else:
                        break
                    for eachload_paths in gen_load_all_paths:
                        if ((available_gen[available_gen['bus'] == eachload_paths.get('gen')].any().any())
                                & (eachload_paths.get('bus') == current_load)):
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
                                if unprocessed_gen_set.intersection(single_path_set):
                                    valid_path_flag = False
                                if valid_path_flag:
                                    if unprocessed_load_set.intersection(single_path_set_load):
                                        valid_path_flag = False
                                if valid_path_flag:
                                    if trans_hv.intersection(single_path_set):
                                        valid_path_flag = False
                                if trans_lv.intersection(single_path_set):
                                    valid_path_flag = False

                                if valid_path_flag == True:
                                    valid_path = all_paths_arr[i]
                                    break

                            if valid_path_flag == False:
                                for i in range(0, len(all_paths_arr)):
                                    valid_path_flag = True
                                    single_path = all_paths_arr[i]
                                    single_path_set = set(single_path)
                                    single_path_set_load = set(single_path[:-1])
                                    if unprocessed_gen_set.intersection(single_path_set):
                                        valid_path_flag = False
                                    if valid_path_flag == True:
                                        if unprocessed_load_set.intersection(single_path_set_load):
                                            valid_path_flag = False
                                    if valid_path_flag == True:
                                        for j in range(0, len(single_path) - 1):
                                            if (net_copy.trafo.loc[
                                                (net_copy.trafo.hv_bus == single_path[j]) &
                                                (net_copy.trafo.lv_bus == single_path[j + 1]) &
                                                (net_copy.trafo.tap_side == 'hv')].any().any()
                                                    or
                                                    net_copy.trafo.loc[
                                                        (net_copy.trafo.hv_bus == single_path[j + 1]) &
                                                        (net_copy.trafo.lv_bus == single_path[j]) &
                                                        (net_copy.trafo.tap_side == 'lv')].any().any()):
                                                valid_path_flag = False
                                                break
                                    if valid_path_flag == True:
                                        valid_path = all_paths_arr[i]
                                        break

                            if valid_path_flag == False:
                                continue

                            if valid_path_flag == True:
                                for i in range(0, len(valid_path) - 1):
                                    if (valid_path[i] is not None) & (valid_path[i + 1] is not None):
                                        line_bw_buses = net_copy.line.loc[(net_copy.line['from_bus'] ==
                                                                           valid_path[i]) & (
                                                                                  net_copy.line['to_bus'] ==
                                                                                  valid_path[i + 1])]
                                        if len(line_bw_buses) == 0:
                                            line_bw_buses = net_copy.line.loc[
                                                (net_copy.line['from_bus'] == valid_path[i + 1]) & (
                                                        net_copy.line['to_bus'] == valid_path[i])]
                                        if len(line_bw_buses) > 0:
                                            net_copy.switch.loc[
                                                (net_copy.switch['element'] == line_bw_buses.index[0]) &
                                                (net_copy.switch['et'] == 'l'), 'closed'] = True
                                        trafo_bw_buses = net_copy.trafo.loc[(net_copy.trafo['hv_bus'] ==
                                                                             valid_path[i]) &
                                                                            (net_copy.trafo['lv_bus'] == valid_path[
                                                                                i + 1])]
                                        if len(trafo_bw_buses) == 0:
                                            trafo_bw_buses = net_copy.trafo.loc[
                                                (net_copy.trafo['lv_bus'] == valid_path[i]) &
                                                (net_copy.trafo['hv_bus'] == valid_path[i + 1])]
                                        if len(trafo_bw_buses) > 0:
                                            net_copy.switch.loc[
                                                (net_copy.switch['element'] == int(trafo_bw_buses.index[0])) &
                                                (net_copy.switch['et'] == 't'), 'closed'] = True
                                        temp_trafo_switch = net_copy.trafo.loc[
                                            net_copy.trafo.hv_bus == valid_path[i]]

                                        if len(temp_trafo_switch) == 0: temp_trafo_switch = net_copy.trafo.loc[
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
                                                                net_copy.trafo.lv_bus == ts_row['lv_bus'])].index.values
                                                    net_copy.switch.loc[
                                                        (net_copy.switch['element'] == trans_index[0]) & (
                                                                net_copy.switch['et'] == 't'), 'closed'] = True
                                unprocessed_load.drop(
                                    unprocessed_load[unprocessed_load['bus'] == int(current_load)].index, inplace=True)
                                try:
                                    motor = sorted_motor[
                                        (sorted_motor.load_bus == int(current_load)) &
                                        (sorted_motor.processed == 'N') &
                                        (sorted_motor.p_inrush_tot < eff_gen_cap) &
                                        (sorted_motor.q_inrush_tot < eff_gen_cap_q)].iloc[0]
                                except IndexError:
                                    motor = None
                                except TypeError:
                                    motor = None
                                    print("test")
                                while 1 == 1:
                                    static = None
                                    motor = None
                                    try:
                                        static = static_data[(static_data.load_bus == int(current_load)) & (
                                                static_data.processed == 'N')].iloc[0]
                                    except IndexError:
                                        static = None
                                    except TypeError:
                                        static = None
                                    if static is not None:
                                        static_p = static['p']
                                        static_q = static['q']
                                    else:
                                        static_p = 0
                                        static_q = 0

                                    try:
                                        motor = sorted_motor[
                                            (sorted_motor.load_bus == int(current_load)) &
                                            (sorted_motor.processed == 'N') &
                                            (np.floor(sorted_motor.p_inrush_tot + static_p) + 5 < math.ceil(
                                                eff_gen_cap)) &
                                            (np.floor(sorted_motor.q_inrush_tot + static_q) + 5 < math.ceil(
                                                eff_gen_cap_q))].iloc[0]
                                    except IndexError:
                                        motor = None
                                    except TypeError:
                                        motor = None
                                    if motor is not None:
                                        net_copy.load.loc[(net_copy.load['bus'] == current_load), 'in_service'] = True
                                        print(iteration)
                                        picked_total_load1 = motor['p_inrush_tot'] + static_p
                                        picked_total_load1_q = motor['q_inrush_tot'] + static_q
                                        net_copy.load.loc[(net_copy.load['bus'] == int(
                                            current_load)), 'p_mw'] = picked_total_load1 + sum(
                                            sorted_motor.loc[(sorted_motor.load_bus ==
                                                              int(current_load)) & (
                                                                         sorted_motor.processed == 'Y'), 'p_total'])
                                        net_copy.load.loc[(net_copy.load['bus'] == int(
                                            current_load)), 'q_mvar'] = picked_total_load1_q + sum(
                                            sorted_motor.loc[(sorted_motor.load_bus ==
                                                              int(current_load)) & (
                                                                     sorted_motor.processed == 'Y'), 'q_total'])
                                        net_copy.load.sn_mva = np.sqrt(
                                            np.power(net_copy.load.p_mw, 2) + np.power(net_copy.load.q_mvar, 2))
                                        picked_steady_load1 = static_p + motor['p_total']
                                        picked_steady_load1_q = static_q + motor['q_total']
                                        random_multi = round(random.uniform(0.05, 0.1), 2)
                                        powerflow_Inrush = pp.runpp(net_copy)
                                        if net_copy.converged == True:

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
                                                            inrush_vd = round(((line_result['vm_hv_pu'] - line_result[
                                                                'vm_lv_pu']) / line_result['vm_hv_pu']) * 100,
                                                                              2).values[0]
                                                        except IndexError:
                                                            inrush_vd = 0
                                                    else:
                                                        line_result = net_copy.res_line.iloc[line_index,]
                                                        try:
                                                            inrush_vd = round(((line_result['vm_from_pu'] -
                                                                                line_result['vm_to_pu']) / line_result[
                                                                                   'vm_to_pu']) * 100, 2).values[0]
                                                        except IndexError:
                                                            inrush_vd = 0

                                                else:
                                                    line_result = net_copy.res_line.iloc[line_index,]
                                                    try:
                                                        inrush_vd = round(((line_result['vm_from_pu'] -
                                                                            line_result['vm_to_pu']) / line_result[
                                                                               'vm_to_pu']) * 100, 2).values[0]
                                                    except IndexError:
                                                        inrush_vd = 0
                                            else:
                                                line_result = net_copy.res_line.iloc[line_index,]
                                                try:
                                                    inrush_vd = round(((line_result['vm_from_pu'] -
                                                                        line_result['vm_to_pu']) / line_result[
                                                                           'vm_to_pu']) * 100, 2).values[0]
                                                except IndexError:
                                                    inrush_vd = 0

                                            net_copy.load.loc[(net_copy.load['bus'] == int(
                                                current_load)), 'q_mvar'] = picked_steady_load1_q + sum(
                                                sorted_motor.loc[(sorted_motor.load_bus ==
                                                                  int(current_load)) & (
                                                                         sorted_motor.processed == 'Y'), 'q_total'])
                                            net_copy.load.sn_mva = np.sqrt(
                                                np.power(net_copy.load.p_mw, 2) + np.power(net_copy.load.q_mvar, 2))
                                            powerflow_Inrush = pp.runpp(net_copy)
                                            iteration = iteration + 1
                                            rest_row = None
                                            rest_df = None
                                            if len(rest_output) > 0:
                                                line_index = net_copy.line[
                                                    (net_copy.line['from_bus'] == valid_path[-2]) &
                                                    (net_copy.line['to_bus'] == valid_path[-1])].index.tolist()
                                                if len(line_index) == 0:
                                                    line_index = net_copy.line[
                                                        (net_copy.line['to_bus'] == valid_path[-2]) &
                                                        (net_copy.line['from_bus'] == valid_path[
                                                            -1])].index.tolist()
                                                    if len(line_index) == 0:
                                                        line_index = net_copy.trafo[
                                                            (net_copy.trafo['lv_bus'] == valid_path[-2]) & (
                                                                    net_copy.trafo['hv_bus'] == valid_path[
                                                                -1])].index.tolist()
                                                        if len(line_index) == 0:
                                                            line_index = net_copy.trafo[
                                                                (net_copy.trafo['hv_bus'] == valid_path[-2]) & (
                                                                        net_copy.trafo['lv_bus'] == valid_path[
                                                                    -1])].index.tolist()
                                                            line_result = net_copy.res_trafo.iloc[line_index,]
                                                            try:
                                                                normalvd = round(((line_result['vm_hv_pu'] -
                                                                                   line_result['vm_lv_pu']) /
                                                                                  line_result['vm_hv_pu']) * 100,
                                                                                 2).values[0]
                                                            except IndexError:
                                                                normalvd = 0
                                                        else:
                                                            line_result = net_copy.res_trafo.iloc[line_index,]
                                                            try:
                                                                normalvd = round(((line_result['vm_lv_pu'] -
                                                                                   line_result['vm_hv_pu']) /
                                                                                  line_result['vm_lv_pu']) * 100,
                                                                                 2).values[0]
                                                            except IndexError:
                                                                normalvd = 0
                                                    else:
                                                        line_result = net_copy.res_line.iloc[line_index,]
                                                        try:
                                                            normalvd = round(((line_result['vm_to_pu'] -
                                                                               line_result['vm_from_pu']) /
                                                                              line_result['vm_from_pu']) * 100,
                                                                             2).values[0]
                                                        except IndexError:
                                                            normalvd = 0
                                                else:

                                                    line_result = net_copy.res_line.iloc[line_index,]
                                                    try:
                                                        normalvd = round(((line_result['vm_from_pu'] -
                                                                           line_result['vm_to_pu']) / line_result[
                                                                              'vm_to_pu']) * 100, 2).values[0]
                                                    except IndexError:
                                                        normalvd = 0
                                                if rest_output.loc[
                                                    rest_output['cranking_power_provided_gen'] ==
                                                    str(c_pow.loc[c_pow['bus'] == next_gen, 'gen_name'].tolist()).strip(
                                                        '[]')].any().any():
                                                    rest_row = [[iteration,
                                                                 '-',
                                                                 round(eff_gen_cap, 2),
                                                                 round(eff_gen_cap_q, 2),
                                                                 '-',
                                                                 '-',
                                                                 '-',
                                                                 net_copy.load.loc[(net_copy.load.bus == int(
                                                                     current_load)), 'name'].values[0],
                                                                 motor['motor'],
                                                                 round(motor['p_inrush_tot'], 2),
                                                                 round(motor['q_inrush_tot'], 2),
                                                                 round(picked_total_load1, 2),
                                                                 round(picked_total_load1_q, 2),
                                                                 round(picked_steady_load1, 2),
                                                                 round(picked_steady_load1_q, 2),
                                                                 round(picked_steady_load1 + (
                                                                         static_p * random_multi), 2),
                                                                 round(picked_steady_load1_q + (
                                                                         static_q * random_multi), 2),
                                                                 inrush_vd,
                                                                 normalvd]]
                                                    rest_df = pd.DataFrame(rest_row, columns=rest_col_names)
                                                else:
                                                    rest_row = [[iteration,
                                                                 str(net_copy.gen.loc[(
                                                                                                  net_copy.gen.in_service == True), 'name'].tolist()).strip(
                                                                     '[]'),
                                                                 round(eff_gen_cap, 2),
                                                                 round(eff_gen_cap_q, 2),
                                                                 '-' if next_gen is None else
                                                                 str(c_pow.loc[c_pow[
                                                                                   'bus'] == next_gen, 'gen_name'].tolist()).strip(
                                                                     '[]'),
                                                                 round(cranking_power, 2),
                                                                 round(cranking_power_q, 2),
                                                                 net_copy.load.loc[(net_copy.load.bus == int(
                                                                     current_load)), 'name'].values[0],
                                                                 motor['motor'],
                                                                 round(motor['p_inrush_tot'], 2),
                                                                 round(motor['q_inrush_tot'], 2),
                                                                 round(picked_total_load1, 2),
                                                                 round(picked_total_load1_q, 2),
                                                                 round(picked_steady_load1, 2),
                                                                 round(picked_steady_load1_q, 2),
                                                                 round(picked_steady_load1 + (
                                                                         static_p * random_multi), 2),
                                                                 round(picked_steady_load1_q + (
                                                                         static_q * random_multi), 2),
                                                                 inrush_vd,
                                                                 normalvd]]
                                                    rest_df = pd.DataFrame(rest_row, columns=rest_col_names)
                                            else:
                                                line_index = net_copy.line[
                                                    (net_copy.line['from_bus'] == valid_path[-2]) &
                                                    (net_copy.line['to_bus'] == valid_path[
                                                        -1])].index.tolist()
                                                if len(line_index) == 0:
                                                    line_index = net_copy.line[
                                                        (net_copy.line['to_bus'] == valid_path[-2]) &
                                                        (net_copy.line['from_bus'] == valid_path[
                                                            -1])].index.tolist()
                                                    if len(line_index) == 0:
                                                        line_index = net_copy.trafo[
                                                            (net_copy.trafo['lv_bus'] == valid_path[-2]) &
                                                            (net_copy.trafo['hv_bus'] == valid_path[
                                                                -1])].index.tolist()
                                                        if len(line_index) == 0:
                                                            line_index = net_copy.trafo[
                                                                (net_copy.trafo['hv_bus'] == valid_path[-2]) &
                                                                (net_copy.trafo['lv_bus'] == valid_path[
                                                                    -1])].index.tolist()
                                                            line_result = net_copy.res_trafo.iloc[line_index,]
                                                            try:
                                                                normalvd = round(((line_result['vm_hv_pu'] -
                                                                                   line_result['vm_lv_pu']) /
                                                                                  line_result[
                                                                                      'vm_hv_pu']) * 100, 2).values[0]
                                                            except IndexError:
                                                                normalvd = 0
                                                        else:
                                                            line_result = net_copy.res_trafo.iloc[line_index,]
                                                            try:
                                                                normalvd = round(((line_result['vm_lv_pu'] -
                                                                                   line_result['vm_hv_pu']) /
                                                                                  line_result['vm_lv_pu']) * 100,
                                                                                 2).values[0]
                                                            except IndexError:
                                                                normalvd = 0
                                                    else:
                                                        line_result = net_copy.res_line.iloc[line_index,]
                                                        try:
                                                            normalvd = round(((line_result['vm_to_pu'] -
                                                                               line_result['vm_from_pu']) /
                                                                              line_result['vm_from_pu']) * 100,
                                                                             2).values[0]
                                                        except IndexError:
                                                            normalvd = 0
                                                else:

                                                    line_result = net_copy.res_line.iloc[line_index,]
                                                    try:
                                                        normalvd = round(((line_result['vm_from_pu'] -
                                                                           line_result['vm_to_pu']) / line_result[
                                                                              'vm_to_pu']) * 100, 2).values[0]
                                                    except IndexError:
                                                        normalvd = 0

                                                rest_row = [[iteration,
                                                             str(net_copy.gen.loc[(
                                                                                              net_copy.gen.in_service == True), 'name'].tolist()).strip(
                                                                 '[]'),
                                                             round(eff_gen_cap, 2),
                                                             round(eff_gen_cap_q, 2),
                                                             str(c_pow.loc[c_pow[
                                                                               'bus'] == next_gen, 'gen_name'].tolist()).strip(
                                                                 '[]'),
                                                             round(cranking_power, 2),
                                                             round(cranking_power_q, 2),
                                                             net_copy.load.loc[(net_copy.load.bus == int(
                                                                 current_load)), 'name'].values[0],
                                                             motor['motor'],
                                                             round(motor['p_inrush_tot'], 2),
                                                             round(motor['q_inrush_tot'], 2),
                                                             round(picked_total_load1, 2),
                                                             round(picked_total_load1_q, 2),
                                                             round(picked_steady_load1, 2),
                                                             round(picked_steady_load1_q, 2),
                                                             round(picked_steady_load1 + (static_p * random_multi), 2),
                                                             round(picked_steady_load1_q + (static_q * random_multi),
                                                                   2),
                                                             inrush_vd,
                                                             normalvd]]
                                                rest_df = pd.DataFrame(rest_row, columns=rest_col_names)
                                            try:
                                                rest_output = rest_output.append(rest_df, ignore_index=False)
                                            except:
                                                rest_output = rest_df.copy()

                                        net_copy.load.loc[(net_copy.load['bus'] == int(
                                            current_load)), 'p_mw'] = picked_steady_load1

                                        net_copy.load.loc[(net_copy.load['bus'] == int(
                                            current_load)), 'q_mvar'] = picked_steady_load1_q
                                        sorted_motor.loc[(sorted_motor['load_bus'] == int(current_load)) &
                                                         (sorted_motor['motor'] == motor[
                                                             'motor']), 'processed'] = 'Y'
                                        powerflow_Inrush = pp.runpp(net_copy)

                                        if static is not None:
                                            static_data.loc[(static_data['load_bus'] == int(current_load)) &
                                                            (static_data['id'] == static['id']), 'processed'] = 'Y'
                                            static_data.loc[(static_data['load_bus'] == int(current_load)) &
                                                            (static_data['id'] == static['id']), 'p'] = static_p + (
                                                    static_p * random_multi)
                                            static_data.loc[(static_data['load_bus'] == int(current_load)) &
                                                            (static_data['id'] == static['id']), 'q'] = static_q + (
                                                    static_q * random_multi)
                                        cranking_power = abs(c_pow.loc[c_pow['bus'] == next_gen, 'pow'].sum())
                                        cranking_power_q = abs(c_pow.loc[c_pow['bus'] == next_gen, 'pow_q'].sum())
                                        try:
                                            processed_load_steadystate_p = static_data.query("processed == 'Y'")[
                                                'p'].sum()
                                            processed_load_steadystate_q = static_data.query("processed == 'Y'")[
                                                'q'].sum()
                                        except IndexError:
                                            processed_load_steadystate_p = 0
                                            processed_load_steadystate_q = 0

                                        try:
                                            processed_load_steadystate_mot_p = sorted_motor.query("processed == 'Y'")[
                                                'p_total'].sum()
                                            processed_load_steadystate_mot_q = sorted_motor.query("processed == 'Y'")[
                                                'q_total'].sum()
                                        except IndexError:
                                            processed_load_steadystate_mot_p = 0
                                            processed_load_steadystate_mot_q = 0

                                        eff_gen_cap = gen_capacity - cranking_power - processed_load_steadystate_p - processed_load_steadystate_mot_p
                                        eff_gen_cap_q = gen_capacity_q - cranking_power_q - processed_load_steadystate_q - processed_load_steadystate_mot_q

                                    else:
                                        if not ((sorted_motor['load_bus'] == int(current_load)) &
                                                (sorted_motor['processed'] == 'N')).any():
                                            load_priority.loc[
                                                load_priority['load_bus'] == int(current_load), 'processed'] = 'Y'
                                            current_load_completed = True
                                            not_completed_load.drop(
                                                not_completed_load[
                                                    not_completed_load['bus'] == int(current_load)].index,
                                                inplace=True)
                                            load_priority.drop(
                                                load_priority[load_priority['load_bus'] == int(current_load)].index,
                                                inplace=True)
                                            net_copy.load.loc[net_copy.load['bus'] == int(current_load), 'p_mw'] = \
                                            net.load.loc[net.load['bus'] == int(current_load), 'p_mw']
                                            net_copy.load.loc[net_copy.load['bus'] == int(current_load), 'q_mvar'] = \
                                            net.load.loc[net.load['bus'] == int(current_load), 'q_mvar']
                                        else:
                                            insufficient_capacity = True

                                        break  # 请问这里break有什么意义？
                            total_static_p = 0
                            total_static_q = 0
                            break
    abs_path = []
    for x in short_path['path']:
        for i, row in net.gen.iterrows():
            if row['bus'] == x:
                if len(abs_path) == 0:
                    abs_path.append(net.ext_grid['name'].iloc[0])
                if row['name'] != net.ext_grid['name'].iloc[0]:
                    abs_path.append(row['name'])
    return rest_output, abs_path


