from multiprocessing import Process, Pool
import multiprocessing
import time

import json
import argparse

from tqdm import tqdm

import os

import subprocess

import time
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--plot', action='store_true', default=False)
    return parser.parse_args()

def run_process_sim_with_timeout(func, args, timeout):
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
        return queue.get()  # 获取返回值
    else:
        return {
            'sim_ret': {'error': 'Function did not return anything'},
            'valid': False,
            'has_zero_resistor': False
        }

SPICE_TYPES = [
    'V', 'I', 'R', 'C', 'L', 'G', 'E', 'F', 'H'
]
def preprocess_spice(spice_code):
    if  not spice_code.startswith('.title'):
        spice_code = '.title ' + spice_code

    # add component id
    comp_num_map = {k: set() for k in SPICE_TYPES}
    for line in spice_code.split('\n'):
        try:
            comp_type = line.split()[0]
            assert comp_type[0] in SPICE_TYPES
        except Exception as e:
            # print(f"Error: {e} when processing line: {line}")
            # new_spice_code += line + '\n'  
            continue
        if len(comp_type) > 1: # add occupied component id
            _type, _id = comp_type[0], comp_type[1:]
            print(f"Add occupied component: {_type}, {_id}")
            comp_num_map[_type].add(str(_id))

    new_spice_code = ''
    cnt = {}
    for line in spice_code.split('\n'):
        try:
            comp_type = line.split()[0]
            assert comp_type[0] in SPICE_TYPES
        except Exception as e:
            print(f"skip line: {line}")
            new_spice_code += line + '\n'
            continue
        if len(comp_type) == 1: # No component id
            print(f"process line: {line}, cnt: {cnt}, map: {comp_num_map}")
            if comp_type not in cnt.keys():
                cnt[comp_type] = 1

            while str(cnt[comp_type]) in comp_num_map[comp_type]:
                cnt[comp_type] += 1 # find a new component id

            line = comp_type + str(cnt[comp_type]) + ' ' + line[1:]
            comp_num_map[comp_type].add(str(cnt[comp_type]))
            print("update map: ", comp_num_map)
        new_spice_code += line + '\n'

    # deal with the old version of control card in spice code
    new_spice_code = new_spice_code.replace('.OP', '.control\nop\n').replace('.END', '.endc\n.end').replace('.PRINT DC', 'print').replace('V(0, ', '-v(').replace(', 0)', ')').replace('V(0 ', '-v(').replace(' 0)', ')').replace(' * ', ' ; ')

    print(f"Preprocessed Spice Code: {new_spice_code}")

    return new_spice_code

def process_single_simulation(qid, spice_code, queue):
    print('\n\n' + '-'*20 + f'Running simulation ... ' + '-'*20 + '\n\n')
    print(f"Raw Spice Code: {spice_code}\n")

    spice_code = preprocess_spice(spice_code)
    print(f"Preprocessed Spice Code: {spice_code}\n")
    
    qid = qid.replace(' ', '_').replace('(', '_').replace(')', '_')
    # try:
        # circuit, sim_ret = spice_to_pyspice(spice_code, require_simulation=True)
    with open(f'./{qid}.sp', 'w') as f:
        f.write(spice_code)
    assert not os.path.exists('output.txt'), "output.txt already exists"
    
    script_path = './utils/simulation/auto_spice.py'
    args = [script_path, '--circuit', f'{qid}.sp', '--output', f'{qid}_ret.json']
    result = subprocess.run(['python3'] + args, check=True, text=True, capture_output=True)
    print(result)

    output = result.stdout
    err_output = result.stderr
    return_code = result.returncode
    print(f"Return Code: {return_code}")
    print(f"Output: {output}")
    print(f"Error Output: {err_output}")

    assert return_code == 0, "Error when running simulation"
    with open(f'{qid}_ret.json', 'r', encoding='utf-8') as f:
        sim_ret = json.load(f)

    print('\n\n' + '## Successfully Run Simulation !! ##' + '\n\n')
    # except Exception as e:
        # print(f"Error: {e} when simulating")
        # sim_ret = {'error:': f'Error when simulating --> {e}'}
        # os.system(f'rm {qid}.sp {qid}_ret.json output.txt')

    os.system(f'rm {qid}.sp {qid}_ret.json output.txt')
    assert not os.path.exists(f'{qid}.sp'), f"File {qid}.sp still exists"
    assert not os.path.exists(f'{qid}_ret.json'), f"File {qid}_ret.json still exists"
    assert not os.path.exists('output.txt'), "output.txt still exists"
    
    print(f"Simulation Results: {sim_ret}")

    ret = {
        'sim_ret': sim_ret,
        'spice_code': spice_code,
    }
    queue.put(ret)
    print('\n\n' + '-'*50)

    return ret

def process_data(data):
    for item in data:
        item['spice'] = item['spice'].replace('-1', '')
    return data

def main():
    args = parse_args()

    file_path = args.path
    assert "step1" in file_path, "File path should contain 'step1'"

    save_path = file_path.replace('.json', '_sim.json')
    done_ids = []
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                done_ids.append(item['id'])

    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]

    data = process_data(data)

    ## NOTE: below is the version of single simulation
    timeout = 5

    results = []
    for item in tqdm(data):
        if item['id'] in done_ids:
            print(f"Skip simulation for {item['id']}... which has been done.")
            continue
        ret_item = {**item}
        print(f">>Running simulation for {item['id']}\n SPIECE: {item['spice']}")
        ret_sim = run_process_sim_with_timeout(func=process_single_simulation, args=(item['id'], item['spice'],), timeout=timeout)
        print(f">>Simulation results: {ret_sim}\n\n\n\n\n\n\n\n")
        # exit()
        print('\n\n' + '-'*50)

        time.sleep(1.)
        
        ret_item.update(ret_sim)
        results.append(ret_item)
    
    print("All tasks completed or timed out.")

    # print(results[0].keys())

    print(f"Saving results to {save_path}")
    with open(save_path, 'a+', encoding='utf-8') as f:
        for item in results:
            try:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f"Error: {e} when writing to file\nitem: {item}")
        # json.dump(results, f, indent=4)

def debug_preprocess():
    spice_code = """
V1 1 0 1
R 1 2 1
R4 2 0 2
R  2 0 3
R2 2 0 4
I 2 0 1

.control
op
print v(1,0)
.endc
.end
    """

    new_spice_code = preprocess_spice(spice_code)
    print(f"New Spice Code: {new_spice_code}")
    pass

if __name__ == '__main__':
    main()
    # debug_preprocess()