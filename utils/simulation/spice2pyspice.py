from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
import re
import numpy as np

# def check_if_all_resistors_are_zero(circuit):
#     # 遍历电路中的所有元件
#     for element in circuit.elements:
#         # 检查元件是否为电阻
#         if isinstance(element, Resistor):
#             # 检查电阻值是否为0
#             if element.resistance != 0@u_Ω:
#                 return False  # 如果发现非0电阻，返回False
#     return True  # 所有电阻都为0，返回True

def parse_unit_string(value_str, unit_type='R'):
    # 假设字符串格式是 "<value><unit>", 例如 "1m" 意味着 1 mA
    try:
        value = float(value_str)
        unit = '1'
    except:
        value = float(value_str[:-1])  # 获取数值部分，转换为浮点数
        unit = value_str[-1]  # 获取最后一个字符作为单位

    unit_dict = {
        'R': {
            '1': u_Ω,
            # 'u': u_uΩ,
            'm': u_mΩ,
            'k': u_kΩ,
        },
        'I': {
            '1': u_A,
            # 'u': u_uA,
            'm': u_mA,
            'k': u_kA,
        },
        'V': {
            '1': u_V,
            'm': u_mV,
            # 'u': u_uV,
            'k': u_kV,
        },
    }

    # 根据单位指定 PySpice 单位
    try:
        unit = unit_dict[unit_type][unit]
        ret = value@unit
    except:
        ret = value

    return ret

simtype2pattern = {
    "op": {
        "current_measure": r'V\((\w+), (\w+)\) / R(\w+) \* measurement of (\w+)', 
        "voltage_measure": r'V\((\w+), (\w+)\) \* measurement of (\w+)',
        "current_measure_2": r"I\((\w+)\)",
    },
}

def has_zero_resistor(circuit):
    for element in circuit.elements:
        if element.name[0] == 'R' and element.resistance <= 1e-6@u_Ω:
            return True
    return False

def spice_to_pyspice(spice, require_simulation=False):
    # spice = spice.replace(' 0 ', ' Z ') # replace '0' with 'Z' to avoid node 0 issue
    if ' 0 ' in spice:
        has_zero_node = True
    else:
        has_zero_node = False
    
    print(f"spice: {spice}")
    # Initialize the circuit
    circuit = Circuit("")
    element_counter = {'R': 0, 'I': 0, 'V': 0}  # Counter for generating element labels

    # Split into lines
    lines = spice.split("\n")

    if require_simulation:
        simulation_result = {
            }
        analysis = None

    # Process each line
    for i, line in enumerate(lines):
        # Remove whitespace and line breaks
        line = line.strip()

        if line.startswith(".title"):
            real_title = line.replace(".title", "").strip()
            circuit.title = real_title
            continue

        # Check if it's an empty line or end statement. If so, skip it.
        if not line or line.upper() == ".END":
            continue

        if require_simulation:
            # try:
            if True:
                if line.startswith(".OP"):
                    simulation_result['type'] = 'op'
                    simulator = circuit.simulator(temperature=25, nominal_temperature=25)

                    if has_zero_resistor(circuit):
                        return circuit, {'error': 'Zero resistor detected in the circuit. Simulation aborted.'}

                    analysis = simulator.operating_point()
                    # if '0' not in analysis.nodes:
                    # node_voltages = {'0': 0.0}
                    node_voltages = {}
                    for node_name, node_value in analysis.nodes.items():
                        node_value = float(node_value[0])
                        node_voltages[node_name] = node_value
                    print(f'nodes: {analysis.nodes}')
                    # if '0' in analysis.nodes:
                    if has_zero_node:
                        node_voltages['0'] = 0.0
                    analysis.node_voltage = node_voltages
                    print('\nnode_voltages: ', node_voltages)

                    min_value = min(node_voltages.values())
                    sorted_nv = sorted(node_voltages.values())
                    node_values = [float(x) - float(min_value) for x in sorted_nv]

                    node_values = np.array(node_values)
                    residual = [1] + [node_values[i] - node_values[i-1] for i in range(1, len(node_values))]
                    idxs_save = [i for i in range(len(residual)) if residual[i] > 1e-6]
                    node_values = node_values[idxs_save]
                    simulation_result['node_voltages'] = list(node_values)
                    for node, value in node_voltages.items():
                        simulation_result[f'node_voltage_{node}'] = value
                    continue

                elif line.startswith(".PRINT"):
                    print(f"Processing line {i}: {line} when simulating ...")

                    for key, pattern in simtype2pattern['op'].items():
                        match = re.search(pattern, line)
                        if match:
                            if key == "current_measure":
                                print(f"pattern {key} got matched")
                                # V(n1, n2) / R1 * measurement of I1
                                n1, n2, r, i = match.groups()
                                n1, n2, r, i = n1.lower(), n2.lower(), r.lower(), i.lower()
                                print(f"matched groups: {n1}, {n2}, {r}, {i}")
                                # simulation_result['current_measure'] = f"I({i}) = V({n1}, {n2}) / R({r})"
                                for element in circuit.elements:
                                    if element.name == f"R{r}":
                                        r_value = element.resistance
                                        break
                                I = (analysis.node_voltage[n1] - analysis.node_voltage[n2]) / r_value

                                simulation_result[f'{i}'.upper()] = I
                                
                            elif key == "voltage_measure":
                                print(f"pattern {key} got matched")
                                # V(n1, n2) * measurement of U1
                                n1, n2, u = match.groups()
                                
                                U = analysis.node_voltage[n1] - analysis.node_voltage[n2]

                                simulation_result[f'{u}'.upper()] = U

                            elif key == "current_measure_2":
                                print(f"pattern {key} got matched")
                                # I(VI1)
                                meas_labl = match.groups()[0]
                                # I = simulator.get_current(analysis, meas_labl)
                                print(analysis.branches)
                                I = analysis.branches[meas_labl.lower()]
                                I_labl = meas_labl[1:]
                                simulation_result[f'{I_labl}'.upper()] = float(I)
                            break

                    continue
            # except Exception as e:
                # print(f"Error parsing line {i}: {line} when simulating ... --> {e}\n\n")
        
        try:
            # Split words to parse the element
            # Assume each element description is "element-name node1 node2 value"
            words = line.split()

            # Get the type of the element (R, I, etc)
            element_type = words[0][0]
            # element_counter[element_type] += 1
            element_label = words[0][1:]

            n1, n2, value = words[1], words[2], words[3]

            # Add the element to the circuit
            value = parse_unit_string(value, element_type.upper())
            if element_type.upper() == "R":
                circuit.R(element_label, n1, n2, value)
            elif element_type.upper() == "I":
                circuit.I(element_label, n1, n2, value)
            elif element_type.upper() == "V":
                circuit.V(element_label, n1, n2, value)

        except Exception as e:
            print(f"Error parsing line {i}: {line} --> {e}")
            continue

    if require_simulation:
        return circuit, simulation_result
    else: 
        return circuit

spice = """
.title Active DC Circuit
R1 N1 N4 85
R4 N1 N2 43
R8 N2 N3 6
R5 N3 N6 9
R3 N4 N5 9
R6 N5 N8 73
R2 N6 N9 27
R7 N8 N9 17
V2 N2 N5 87
V1 N7 N8 72
I1 N4 N7 48

.END
"""

spice_sim = """
.title Active DC Circuit
R2 0 1 10
R1 1 0 10
R3 1 2 5
R6 2 5 5
R5 2 3 5
R4 3 5 5
V1 0 4 5 * U_{S1}
I1 4 5 5


.OP
.PRINT DC V(1, 2) / R3 * measurement of I30 : I(R3)
.PRINT DC V(2, 5) * measurement of U57
.PRINT DC V(2, 3) / R5 * measurement of I76 : I(R5)
.END
"""

spice_sim_2 = """
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
.PRINT DC V(N1, N5) / R3 * measurement of I : I(R3)
.END
"""

spice_sim_cur_meas_2 = """
.title Active DC Circuit
R1 N1 N2 4k
R2 N3 N2 4k
R3 N5 NR3 2k
VI NR3 N1 0
R4 N3 N5 3k
VS1 N1 N3 25
IS1 N3 N2 3m
IS2 N5 N1 10m
IS3 N5 N2 5m

.OP
.PRINT DC I(VI) * measurement of I
.END
"""

def debug():
    print('\n' + '-'*50 + '\n')

    circuit = spice_to_pyspice(spice)
    print(f"Circuit: {circuit}")

    print('\n' + '-'*50 + '\n')

    circuit_sim, sim_ret = spice_to_pyspice(spice_sim, require_simulation=True)
    print(f"Circuit_Sim: {circuit_sim}\nSimulation Result: {sim_ret}")

    print('\n' + '-'*50 + '\n')

def debug_re():
    pattern = r'V\((\w+), (\w+)\) / R(\w+) \* measurement of I(\w+)'
    stringtest = ".PRINT DC V(1, 5) / R3 * measurement of I30 : I(R3)"
    match = re.search(pattern, stringtest)
    print(match.groups())

def debug_0524():
    spice_str = """
.title Active DC Circuit
R1 1 0 10
VS1 2 0 20
R2 3 2 4
R3 4 0 8
R4 1 3 10
R5 3 4 8
R6 1 5 2
VS2 5 4 40

.OP
.END"""
#     spice_str = """
# .title Active DC Circuit
# R1 1 2 2
# R4 5 7 2
# R3 2 3 10
# R5 6 7 1
# R2 7 8 20
# V2 6 1 9
# V3 5 2 18
# V1 8 3 30


# .OP
# .END
# """
    spice_str_2 = """.title Active DC Circuit
R1 1 2 1
R2 2 3 20
R3 2 5 2
R4 4 6 2
R5 6 0 10
V1 1 4 9
V2 5 6 18
V3 3 0 30

`
.OP
.END"""
    circuit, sim_ret = spice_to_pyspice(spice_str, require_simulation=True)
    print(f"Circuit: {circuit}\nSimulation Result: {sim_ret}")
    # circuit_2, sim_ret_2 = spice_to_pyspice(spice_str_2, require_simulation=True)
    # print(f"Circuit: {circuit_2}\nSimulation Result: {sim_ret_2}")

def debug_0_node():
    from PySpice.Spice.Netlist import Circuit

    # 创建一个电路
    circuit = Circuit('Sample Circuit')

    # 添加一些元件
    circuit.V('1', '1', '0', 10)  # 电压源从节点1到节点0（地）
    circuit.R('1', '1', '2', 1000)  # 1k欧姆电阻从节点1到节点2

    # 检查是否有节点命名为 '0'
    if '0' in circuit.nodes:
        print("Node 0 is present in the circuit.")
    else:
        print("No node 0 found in the circuit.")

def debug_0528():
    circ, sim_ret = spice_to_pyspice(spice_sim_cur_meas_2, require_simulation=True)

    print(f"Circuit: {circ}\nSimulation Result: {sim_ret}")

if __name__ == '__main__':
    # debug()
    # debug_re()
    # debug_0524()
    # debug_0_node()
    debug_0528()