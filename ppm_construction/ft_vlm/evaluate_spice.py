import os
import argparse
# from functools import partial
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
# from utils.simulation.spice_utils import similarity_by_simulation_node_voltage
import concurrent.futures

from utils.simulation.spice2pyspice import spice_to_pyspice
from utils.simulation.spice_utils import get_components_stat_from_spice, get_node_num_from_spice

from multiprocessing import Process, Pool
import multiprocessing
import time

from tqdm import tqdm

import argparse

import numpy as np
import matplotlib.pyplot as plt


SPICE_TYPES = [
    'V', 'I', 'R', 'C', 'L', 'G', 'E', 'F', 'H'
]

print("Loaded all modules")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./ft_vlm/results/test_minitest_240430_cogagent-vqa-05-23-06-30.json')
    parser.add_argument('--plot', action='store_true', default=False)
    return parser.parse_args()

def float_close(f1, f2, tol=1e-3):
    """Compare whether two floats are equal within a given tolerance."""
    return abs(f1 - f2) < tol

def _compare_values(v1, v2, tol=1e-3, ignore_abs=False, ignore_string=False):
    """Compare two values, which can be floats or lists of floats."""
    if isinstance(v1, str) and isinstance(v2, str):
        if ignore_string:
            return True
        return v1 == v2
    if isinstance(v1, int) and isinstance(v2, int):
        if ignore_abs:
            v1, v2 = abs(v1), abs(v2)
        return v1 == v2
    if isinstance(v1, float) and isinstance(v2, float):
        if ignore_abs:
            v1, v2 = abs(v1), abs(v2)
        return float_close(v1, v2, tol)
    if isinstance(v1, list) and isinstance(v2, list):
        if len(v1) != len(v2):
            return False

        v1 = sorted(v1)
        v2 = sorted(v2)
        for i in range(len(v1)):
            if not _compare_values(v1[i], v2[i], tol, ignore_abs):
                print(f"Values not equal in {i}th element of list: {v1[i]} vs {v2[i]}")
                return False
        return True
    elif isinstance(v1, float) and isinstance(v2, float):
        return float_close(v1, v2, tol)
    else:
        return False
    
def compare_values(v1, v2, tol=1e-3, ignore_abs=False,  ignore_string=False):
    """Compare two values, which can be floats or lists of floats."""
    ans = _compare_values(v1, v2, tol, ignore_abs,  ignore_string=ignore_string)
    # print(f"#"*50)
    return ans

def normalized_dict(d):
    new_d = {}
    for key, value in d.items():
        if isinstance(value, str):
            try:
                new_d[key.lower()] = float(value[:-1])
            except Exception as e:
                new_d[key.lower()] = value
        else:
            new_d[key.lower()] = value
    return new_d

def compare_dicts(d1, d2, tol=1e-3, ignore_abs=False, ignore_string=False):
    d1 = normalized_dict(d1)
    d2 = normalized_dict(d2)
    """Compare all keys and values of two dictionaries."""
    if d1 == {} and d2 == {}:
        return True

    keys_1 = sorted(list(d1.keys()))
    keys_2 = sorted(list(d2.keys()))
    if sorted(keys_1) != sorted(keys_2):
        print(f"Keys not equal: {keys_1} vs {keys_2}")
        return False

    for key in d1:
        if not compare_values(d1[key], d2[key], tol, ignore_abs=ignore_abs,  ignore_string=ignore_string):
            print(f"Values not equal for key {key}: {d1[key]} vs {d2[key]}")
            return False
    return True

def check_equal_measurements(ret_1, ret_2, thres=1e-3, note="", ignore_abs=False, ignore_string=False):
    print('\n\n' + '-'*50)
    remove_keys = []
    for k in ret_1.keys(): 
        if k.startswith('node_voltage_'):
            remove_keys.append(k)
    for k in remove_keys:
        ret_1.pop(k)
    remove_keys = []
    for k in ret_2.keys():
        if k.startswith('node_voltage_'):
            remove_keys.append(k)
    for k in remove_keys:
        ret_2.pop(k)
    print(f"Checking Equal Measurements {note}: {ret_1} vs {ret_2}")
    if ret_1 == {} and ret_2 == {}:
        return False
    return compare_dicts(ret_1, ret_2, thres, ignore_abs=ignore_abs, ignore_string=ignore_string)

def process_task(id, item):
    pred, label = item['pred'], item['label']

    print('\n\n' + '-'*20 + f'Evaluate {id}th data' + '-'*20 + '\n\n')
    print(f"Pred: {pred}\nLabel: {label}\n")

    circuit_pred, circuit_label = None, None
    if not pred.startswith('.title'):
        pred = '.title ' + pred
    if not label.startswith('.title'):
        label = '.title ' + label

    try:
        circuit_label, sim_ret_label = spice_to_pyspice(label, require_simulation=True)
        print('\n\n' + '## Successfully Run Simulation for Label !! ##' + '\n\n')
    except Exception as e:
        print(f"Error: {e} when simulating label")
        sim_ret_label = {}

    try:
        circuit_pred, sim_ret_pred = spice_to_pyspice(pred, require_simulation=True)
        print('\n\n' + '## Successfully Run Simulation for Prediction !! ##' + '\n\n')
    except Exception as e:
        print(f"Error: {e} when simulating prediction")
        sim_ret_pred = {}


    print(f"Simulation Results for Prediction: {sim_ret_pred}")
    print(f"Simulation Results for Label: {sim_ret_label}")
    print('\n\n' + '-'*50)

    return {
        'label_valid': (circuit_label is not None),
        'has_zero_resistor': 'error' in sim_ret_label and 'Zero resistor' in sim_ret_label['error'],
        'pred_valid': (circuit_pred is not None),
        'label_sim_ret': sim_ret_label,
        'pred_sim_ret': sim_ret_pred,
        'label': label,
        'pred': pred,
        'id': id
    }

def preprocess_spice(spice_code):
    if  not spice_code.startswith('.title'):
        spice_code = '.title ' + spice_code

    # add component id
    cnt = {}
    new_spice_code = ''
    for line in spice_code.split('\n'):
        try:
            comp_type = line.split()[0]
            assert comp_type in SPICE_TYPES
        except Exception as e:
            print(f"Error: {e} when processing line: {line}")
            new_spice_code += line + '\n'
            continue
        if len(comp_type) == 1: # No component id
            if comp_type not in cnt.keys():
                cnt[comp_type] = 0
            cnt[comp_type] += 1
            line = comp_type + str(cnt[comp_type]) + ' ' + line[1:]
        new_spice_code += line + '\n'

    # deal with the old version of control card in spice code
    new_spice_code = new_spice_code.replace('.OP', '.control\nop\n').replace('.END', '.endc\n.end').replace('.PRINT DC', 'print').replace('V(0, ', '-v(').replace(', 0)', ')').replace('V(0 ', '-v(').replace(' 0)', ')').replace(' * ', ' ; ')

    print(f"Preprocessed Spice Code: {new_spice_code}")

    return new_spice_code
    pass

def process_single_simulation(spice_code, qid, queue):
    print('\n\n' + '-'*20 + f'Running simulation ... ' + '-'*20 + '\n\n')
    print(f"Spice Code: {spice_code}\n")

    circuit = None
    spice_code = preprocess_spice(spice_code)

    # try:
        # circuit, sim_ret = spice_to_pyspice(spice_code, require_simulation=True)
    with open(f'./{qid}.sp', 'w') as f:
        f.write(spice_code)
    os.system(f'python ./utils/simulation/auto_spice.py --circuit {qid}.sp --output {qid}_ret.json')
    with open(f'{qid}_ret.json', 'r', encoding='utf-8') as f:
        sim_ret = json.load(f)
    print(f"Simulation Results: {sim_ret}")
    # exit()
    os.system(f'rm {qid}.sp {qid}_ret.json')

    print('\n\n' + '## Successfully Run Simulation !! ##' + '\n\n')
    # except Exception as e:
    #     # print(f"Error: {e} when simulating")
    #     print(f'Error when simulating [SPICE CODE]\n{spice_code} \n-->\n [ERROR]\n{e}')
    #     sim_ret = {"error": str(e)}

    os.system(f'rm {qid}.sp {qid}_ret.json')
    
    print(f"Simulation Results: {sim_ret}")

    ret = {
        'valid': (circuit is not None),
        'sim_ret': sim_ret,
        'spice_code': spice_code,
        'has_zero_resistor': 'error' in sim_ret and 'Zero resistor' in sim_ret['error'],
    }
    queue.put(ret)
    print('\n\n' + '-'*50)

    return ret

def run_process_with_timeout(func, args, timeout):
    queue = multiprocessing.Queue()

    p = multiprocessing.Process(target=func, args=(*args, queue))
    p.start()
    p.join(timeout)

    if p.is_alive():
        print("Function is still running, terminating now...")
        p.terminate()
        p.join()

        return {
            'sim_ret': {'error': 'Function timed out'},
            'valid': False,
            'has_zero_resistor': False
        }
    
    if not queue.empty():
        return queue.get()
    else:
        return {
            'sim_ret': {'error': 'Function did not return anything'},
            'valid': False,
            'has_zero_resistor': False
        }

def convert_numstr_to_float(num_str):
    d = {
        'm': 'e-3',
        'u': 'e-6',
        'k': 'e3',
    }
    for k, v in d.items():
        if k in num_str:
            num_str = num_str.replace(k, v)
    # print(f"Converted: {num_str}")
    try:
        ret = float(num_str)
    except Exception as e:
        print(f"Error: {e} when converting numstr to float")
        print(f"Numstr: {num_str}")
        ret = 0
    return ret

def evaluate(result, args):

    print(len(result))

    print('\n\n' + '#'*20 + "components analysis" + '#'*20 + '\n\n')
    for r in result:
        r['comp_stat_label'] = get_components_stat_from_spice(r['label'])
        r['comp_stat_pred'] = get_components_stat_from_spice(r['pred'])

        r['comp_stat_label'] = {k: [convert_numstr_to_float(vv) for vv in v] for k, v in r['comp_stat_label'].items()}
        r['comp_stat_pred'] = {k: [convert_numstr_to_float(vv) for vv in v] for k, v in r['comp_stat_pred'].items()}

    ignore_abs = True
    result = [{**r, 'comp_amnt_label': {k: len(v) for k, v in r['comp_stat_label'].items()}, 'comp_amnt_pred': {k: len(v) for k, v in r['comp_stat_pred'].items()}} for r in result]
    result = [{"comp_amnt_equal": compare_dicts(r['comp_amnt_label'] , r['comp_amnt_pred']), "comp_stat_equal": compare_dicts(r['comp_stat_label'] , r['comp_stat_pred']), **r} for r in result]
    result = [
                {
                    'sim_ret_equal': check_equal_measurements(r['label_sim_ret'], r['pred_sim_ret'], 
                                        note=f"for question {r['qid']}", 
                                        ignore_abs=ignore_abs,
                                        ignore_string=True
                                        ), **r
                } 
            for r in result]

    # 数值型
    num_no_zr = sum([r['label_valid'] and not r['has_zero_resistor'] for r in result])

    num_acc_str_no_zr = sum([r['label_valid'] and not r['has_zero_resistor'] and r['label'].strip() == r['pred'].strip() for r in result])
    num_acc_predvalid_no_zr = sum([r['label_valid'] and not r['has_zero_resistor'] and r['pred_valid'] for r in result])
    num_acc_compamnt_no_zr = sum([r['label_valid'] and not r['has_zero_resistor'] and r['comp_amnt_equal'] for r in result])
    num_acc_compstat_no_zr = sum([r['label_valid'] and not r['has_zero_resistor'] and r['comp_stat_equal'] for r in result])
    num_acc_simret_no_zr = sum([r['label_valid'] and not r['has_zero_resistor'] and r['sim_ret_equal'] for r in result])

    # 标签型
    num_zr = sum([r['label_valid'] and r['has_zero_resistor'] for r in result])
    num_acc_str_zr = sum([r['label_valid'] and r['has_zero_resistor'] and r['label'].strip() == r['pred'].strip() for r in result])
    num_acc_predvalid_zr = sum([r['label_valid'] and r['has_zero_resistor'] and r['pred_valid'] for r in result])
    num_acc_compamnt_zr = sum([r['label_valid'] and r['has_zero_resistor'] and r['comp_amnt_equal'] for r in result])
    num_acc_compstat_zr = sum([r['label_valid'] and r['has_zero_resistor'] and r['comp_stat_equal'] for r in result])
    num_acc_simret_zr = sum([r['label_valid'] and r['has_zero_resistor'] and r['sim_ret_equal'] for r in result])

    if num_no_zr > 0:
        print(f"When Label Valid, Prediction Valid(No Zero Resistor): {num_acc_predvalid_no_zr} / {num_no_zr} = {num_acc_predvalid_no_zr / num_no_zr}")
        print(f"When Label Valid, Prediction Valid, String Equal(No Zero Resistor): {num_acc_str_no_zr} / {num_no_zr} = {num_acc_str_no_zr / num_no_zr}")
        print(f"When Label Valid, Prediction Valid, Component Amount Equal(No Zero Resistor): {num_acc_compamnt_no_zr} / {num_no_zr} = {num_acc_compamnt_no_zr / num_no_zr}")
        print(f"When Label Valid, Prediction Valid, Component Equal(No Zero Resistor): {num_acc_compstat_no_zr} / {num_no_zr} = {num_acc_compstat_no_zr / num_no_zr}")
        print(f"When Label Valid, Prediction Valid, Simulation Result Equal(No Zero Resistor): {num_acc_simret_no_zr} / {num_no_zr} = {num_acc_simret_no_zr / num_no_zr}")
    else:
        print(f"No No Zero Resistor Data")

    if num_zr > 0:
        print(f"When Label Valid, Prediction Valid(Zero Resistor): {num_acc_predvalid_zr} / {num_zr} = {num_acc_predvalid_zr / num_zr}")
        print(f"When Label Valid, Prediction Valid, String Equal(Zero Resistor): {num_acc_str_zr} / {num_zr} = {num_acc_str_zr / num_zr}")
        print(f"When Label Valid, Prediction Valid, Component Amount Equal(Zero Resistor): {num_acc_compamnt_zr} / {num_zr} = {num_acc_compamnt_zr / num_zr}")
        print(f"When Label Valid, Prediction Valid, Component Equal(Zero Resistor): {num_acc_compstat_zr} / {num_zr} = {num_acc_compstat_zr / num_zr}")
        print(f"When Label Valid, Prediction Valid, Simulation Result Equal(Zero Resistor): {num_acc_simret_zr} / {num_zr} = {num_acc_simret_zr / num_zr}")
    else:
        print(f"No Zero Resistor Data")

    compnum2count = {}
    compnum2simretacc = {}
    nodenum2count = {}
    nodenum2simretacc = {}
    for r in result:
        if not r['label_valid'] or r['has_zero_resistor']:
            continue
        compnum = np.sum([len(v) for v in r['comp_stat_label'].values()])
        nodenum = get_node_num_from_spice(r['label'])
        if compnum not in compnum2count.keys():
            compnum2count[compnum] = 0
            compnum2simretacc[compnum] = []
        
        compnum2count[compnum] += 1
        compnum2simretacc[compnum].append(r['sim_ret_equal'])

        if nodenum not in nodenum2count.keys():
            nodenum2count[nodenum] = 0
            nodenum2simretacc[nodenum] = []
        
        nodenum2count[nodenum] += 1
        nodenum2simretacc[nodenum].append(r['sim_ret_equal'])
    
    compnum2simretacc = {k: np.mean(v) for k, v in compnum2simretacc.items()}
    nodenum2simretacc = {k: np.mean(v) for k, v in nodenum2simretacc.items()}

    if args.plot:
        path_dir, path_name = os.path.dirname(args.path), os.path.basename(args.path)
        fig_dir = os.path.join(path_dir, path_name.replace('.json', '') + '_figures')
        os.makedirs(fig_dir, exist_ok=True)

        # plot acc vs compnum
        plt.figure()
        plt.bar(compnum2count.keys(), [compnum2simretacc[k] for k in compnum2count.keys()])
        plt.xlabel('Component Number')
        plt.ylabel('Simulation Accuracy')
        plt.title('Simulation Accuracy vs Component Number')

        plt.savefig(os.path.join(fig_dir, 'sim_acc_vs_compnum.png'))

        # plot acc vs nodenum
        plt.figure()
        plt.bar(nodenum2count.keys(), [nodenum2simretacc[k] for k in nodenum2count.keys()])
        plt.xlabel('Node Number of Netlist')
        plt.ylabel('Simulation Accuracy')
        plt.title('Simulation Accuracy vs Node Number')

        plt.savefig(os.path.join(fig_dir, 'sim_acc_vs_nodenum.png'))

        # plot compnum distribution
        plt.figure()
        plt.bar(compnum2count.keys(), [compnum2count[k] for k in compnum2count.keys()])
        plt.xlabel('Component Number')
        plt.ylabel('Count')
        plt.title('Component Number Distribution')

        plt.savefig(os.path.join(fig_dir, 'compnum_distribution.png'))

        # plot nodenum distribution
        plt.figure()
        plt.bar(nodenum2count.keys(), [nodenum2count[k] for k in nodenum2count.keys()])
        plt.xlabel('Node Number of Netlist')
        plt.ylabel('Count')
        plt.title('Node Number Distribution')

        plt.savefig(os.path.join(fig_dir, 'nodenum_distribution.png'))

        # os.makedirs(fig_dir, exist_ok=True)
    pass

    return result


def multi_process_task(data, task, num_proc=4, timeout=5):
    results = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_proc) as executor:
        futures = [executor.submit(task, i, item) for i, item in enumerate(data)]
        
        for future in concurrent.futures.as_completed(futures, timeout=timeout):
            try:
                result = future.result(timeout=timeout)
                results.append(result)
                print(f"Task completed: {result}")
            except concurrent.futures.TimeoutError:
                print("Task timed out")
                futures.remove(future)
                results.append({
                    'Error': 'Task timed out'
                })
            except Exception as e:
                print(f"Task generated an exception: {e}")
    
    return results

def process_data(data):
    
    for item in data:
        item['pred'] = item['pred'].replace('-1', '')

    return data

def main():
    args = parse_args()
    file_path = args.path

    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]

    data = process_data(data)

    ## NOTE: below is the version of single simulation
    timeout = 5

    results = []
    for item in tqdm(data):
        ret_item = {**item}

        ret_label = run_process_with_timeout(func=process_single_simulation, args=(item['label'], item['qid']), timeout=timeout)
        print(f"Label: {item['label']}")
        print(f"Label Results: {ret_label}")

        ret_item.update(
            {
                'label_valid': ret_label['valid'],
                'label_sim_ret': ret_label['sim_ret'],
                'has_zero_resistor': ret_label['has_zero_resistor']
            }
        )
        ret_pred = run_process_with_timeout(func=process_single_simulation, args=(item['pred'], item['qid']), timeout=timeout)
        
        print(f"Prediction: {item['pred']}")
        print(f"Prediction Results: {ret_pred}")
        ret_item.update(
            {
                'pred_valid': ret_pred['valid'],
                'pred_sim_ret': ret_pred['sim_ret']
            }
        )
        results.append(ret_item)
    
    print("All tasks completed or timed out.")

    results = evaluate(results, args)
    print(results[0].keys())
    save_path = file_path.replace('.json', '_eval.json')
    print(f"Saving results to {save_path}")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()