import os
import sys
sys.path.append(os.path.dirname(__file__))
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

from PySpice.Spice.NgSpice.Shared import NgSpiceShared

ngspice = NgSpiceShared.new_instance() 

# Using PySpice to load circuit from spice file
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

from spice2pyspice import spice_to_pyspice

import numpy as np

import networkx as nx
from itertools import permutations

def check_circuit_valid(spice_code: str) -> bool:
    elements_ = spice_code.strip().split('\n')
    elements = []
    for ele in elements_:
        if ele.startswith('.') or len(ele.strip()) == 0:
            continue
        elements.append(ele)
    G = nx.MultiGraph()
    print(elements)
    
    for el in elements:
        print(el)
        if "*" in el:
            el = el[:el.index("*")]
        ret = el.split()
        assert len(ret) == 4, f"element: {el} is not valid"
        name, n1, n2, _ = ret
        prop = {'current': False, 'voltage': False}
        if name[0] == 'V':
            prop['voltage'] = True
        elif name[0] == 'I':
            prop['current'] = True
        print(f"Adding edge: {n1} -> {n2} with prop: {prop}")
        G.add_edge(n1, n2, **prop)

    print(G.degree())
    print(G.edges())

    # Check for parallel voltage sources
    for n1, n2 in G.edges():
        if sum(G[n1][n2][i]['voltage'] for i in G[n1][n2]) > 1:
            print("More than one voltage source in parallel")
            return False

    # G is your graph
    nodes_with_degree_one_current = [node for node, degree in G.degree() if degree == 1 and any(G[node][nbr][0]['current'] for nbr in G[node])]
    if len(nodes_with_degree_one_current) > 0:
        print(f"Degree one current source, node: {nodes_with_degree_one_current}")
        return False

    # TODO: Check for series current sources using 
    pass
    # for cycle in nx.simple_cycles(G):
    #     print(f"Cycle: {cycle}")
    #     cycle = cycle + [cycle[0]]
    #     current_sources = 0
    #     for i in range(len(cycle) - 1):
    #         current_sources += sum(G[cycle[i]][cycle[i + 1]][j]['current'] for j in G[cycle[i]][cycle[i + 1]])
    #     if current_sources > 1:
    #         print("More than one current source in series")
    #         return False
    # exit()

    return True

def get_node_num_from_spice(spice_code):
    node_num = 0
    nodes = set()
    for line in spice_code.split('\n'):
        if line.startswith('.'):
            continue
        print(line)
        line = line.split('*')[0].strip()
        if len(line) == 0:
            continue
        try:
            comp, n1, n2, val = line.split()
            nodes.add(n1)
            nodes.add(n2)
        except:
            print(f"Error in line: {line}")
            continue

    node_num = len(nodes)
    return node_num

def get_components_stat_from_spice(spice_code):
    comps = []

    for line in spice_code.split('\n'):
        if line.startswith('.'):
            continue
        print(line)
        line = line.split('*')[0].strip()
        if len(line) == 0:
            continue
        try:
            comp, n1, n2, val = line.split()
            if comp.startswith('VI'):
                # comp = comp[2:]
                continue
            comp = comp[0]
            comps.append((comp, val))
        except:
            print(f"Error in line: {line}")
            continue

    comp_types = set([comp for comp, _ in comps])
    comp_stats = {comp: [] for comp in comp_types}
    for comp, val in comps:
        comp_stats[comp].append(val)
    print(f"Components stats: {comp_stats} from spice code: {spice_code}")
    return comp_stats


def get_nodes_voltage(spice_code):
    if not check_circuit_valid(spice_code):
        return []
    
    try: 
        circuit = spice_to_pyspice(spice_code)

        print('Loaded circuit:')
        print(circuit)

        # Run simulation and get result
        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        print(f"Start simulation...")
        analysis = simulator.operating_point()
        print(f"Simulation done!")
        # print(f'Analysis: \n\tnodes: {analysis.nodes} \n\tbranches: {analysis.branches}')

        # current = analysis['R1'].dc_value
        # voltage = analysis['R1'].dc_voltage
        # print(f"current: {current}A, voltage: {voltage}V")
        
        node_values = []
        for node_name, node_value in analysis.nodes.items():
            node_value = float(node_value[0])
            print(f'Node {node_name}: { node_value:.2f}V')
            node_values.append(node_value)

        # node_values = np.unique(node_values)
        node_values = np.array(node_values)
        node_values = np.sort(node_values) - np.min(node_values)

        residual = [1] + [node_values[i] - node_values[i-1] for i in range(1, len(node_values))]
        idxs_save = [i for i in range(len(residual)) if residual[i] > 1e-6]
        node_values = node_values[idxs_save]
    
    except Exception as e:
        print("Error:", e)
        node_values = None
    
    return node_values

def similarity_by_simulation_node_voltage(spice_code_1, spice_code_2, return_node_values=False):
    simi = -1e3
    node_values_1 = get_nodes_voltage(spice_code_1)
    node_values_2 = get_nodes_voltage(spice_code_2)

    if node_values_1 is None or node_values_2 is None:
        return simi

    similarity = min(- np.log(np.linalg.norm(node_values_1 - node_values_2)), 999)
    if not return_node_values:
        return similarity
    else:
        return similarity, node_values_1, node_values_2


spice_1 = """
.title Active DC Circuit
R4 N1 N3 97
R2 N1 N2 93
R3 N4 N3 83
R1 N4 N2 69
I1 N2 N3 46
"""


spice_2 = """
.title Active DC Circuit
R2 N4 N3 83
R4 N4 N2 69
R1 N1 N3 97
R3 N1 N2 93
I1 N2 N3 46
"""


spice_3 = """
.title Active DC Circuit
R4 N5 N3 97
R2 N5 N6 93
R3 N4 N3 83
R1 N4 N6 69
I1 N6 N3 46
"""

spice_4= """
.title Active DC Circuit
R4 N1 N3 97
R2 N1 N2 93
R3 N4 N3 83
R1 N4 N2 72
I1 N2 N3 46
"""

test_spice = """
.title Active DC Circuit
R3 N2 N3 34
R4 N5 N3 67
R2 N3 N4 66
I2 N1 N5 20
I1 N1 N2 14

.END
"""

test_spice_sim = """
.title Active DC Circuit
R1 N1 N2 4k
R2 N3 N2 4k
R3 N1 N5 2k
R4 N3 N5 3k
VS1 N1 N3 25
IS1 N3 N2 3m
IS2 N5 N1 10m
IS3 N5 N2 5m

.OP
.PRINT DC V(1, 5) / R3 * measurement of I30 : I(R3)
.END
"""


def main():
    node_values_list = []
    for i in range(1, 5):
        print(f"Running spice_{i}...")
        spice_code = globals()[f'spice_{i}']
        valid = check_circuit_valid(spice_code)
        print(f"Is circuit valid: {valid}")
        if not valid:
            continue
        node_values = get_nodes_voltage(spice_code)
        node_values_list.append(node_values)
    for i in range(1, 5):
        print(f"spice_{i}\t{node_values_list[i-1]}\n")

def debug(spice_code):
    valid = check_circuit_valid(spice_code)
    print(f"Is circuit valid: {valid}")
    if valid:
        get_nodes_voltage(spice_code)

def debug_get_components_stat_from_spice():
    print(get_components_stat_from_spice(test_spice_sim))

if __name__ == "__main__":
    # main()
    # debug(test_spice)
    # debug(test_spice_sim)
    debug_get_components_stat_from_spice()