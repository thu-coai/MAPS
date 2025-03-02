import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)
import json
import numpy as np
import random

from concurrent.futures import ThreadPoolExecutor
import threading

import argparse
from tqdm import tqdm

from grid_rules import gen_circuit, TYPE_RESISTOR, TYPE_CAPACITOR, TYPE_INDUCTOR, TYPE_VOLTAGE_SOURCE, TYPE_CURRENT_SOURCE, TYPE_VCCS, TYPE_VCVS, TYPE_CCCS, TYPE_CCVS, TYPE_SHORT, MEAS_TYPE_VOLTAGE, MEAS_TYPE_CURRENT

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--note", type=str, default="v4")
    parser.add_argument("--gen_num", type=int, default=1000)
    parser.add_argument("--save_path", type=str, default="./data/grid/test.json")
    parser.add_argument("--num_proc", type=int, default=8)

    args = parser.parse_args()
    return args

def threading_task(task_id, seed, note, gen_num, save_path):
    # seed=42
    np.random.seed(seed)
    random.seed(seed)
    cnt = 0
    for i in tqdm(range(gen_num), desc=f"Task {task_id}"):
        while True:
            id = f"{task_id}_{cnt+1}"
            try:
                circ = gen_circuit(note, id=id)
                latex_code = circ.to_latex()
                spice_code = circ._to_SPICE()
            except Exception as e:
                print(f"Error: {e}")
                circ = None
            if circ and circ.valid:
                break

        print(f"Generating Latex code {id}...")
        # latex_code = circ.to_latex()
        print("\n\n```latex")
        print(latex_code)
        print("```\n\n")

        assert circ is not None and circ.valid
        # if circ and circ.valid:
        with open(save_path.replace(".json", ".txt"), "a+", encoding='utf-8') as file:
            print(f"{id} valid, Saving {id}...", file)
        stat_info = {
            "num_nodes": len(circ.nodes),
            "num_branches": len(circ.branches),
            "num_resistors": len([1 for br in circ.branches if br['type'] == TYPE_RESISTOR]),
            "num_capacitors": len([1 for br in circ.branches if br['type'] == TYPE_CAPACITOR]),
            "num_inductors": len([1 for br in circ.branches if br['type'] == TYPE_INDUCTOR]),
            "num_voltage_sources": len([1 for br in circ.branches if br['type'] == TYPE_VOLTAGE_SOURCE]),
            "num_current_sources": len([1 for br in circ.branches if br['type'] == TYPE_CURRENT_SOURCE]),
            "num_controlled_sources": len([1 for br in circ.branches if br['type'] in [TYPE_VCCS, TYPE_VCVS, TYPE_CCCS, TYPE_CCVS]]),
            "num_shorts": len([1 for br in circ.branches if br['type'] == TYPE_SHORT]),
            "num_voltage_measurements": len([1 for br in circ.branches if br['type'] == TYPE_RESISTOR and br['measure'] == MEAS_TYPE_VOLTAGE]),
            "num_current_measurements": len([1 for br in circ.branches if br['type'] == TYPE_RESISTOR and br['measure'] == MEAS_TYPE_CURRENT]),
        }

        with open(save_path, "a+", encoding='utf-8') as f:
            new_item = {
                "id": id,
                "latex": latex_code,
                "spice": spice_code,
                "stat": stat_info
            }
            f.write(json.dumps(new_item, ensure_ascii=False) + "\n")
        cnt = cnt + 1

def main(args):

    note = args.note
    gen_num = args.gen_num
    save_path = args.save_path
    num_proc = args.num_proc

    use_concurrent = True

    with open(save_path, "w", encoding='utf-8') as f:
        f.write("")

    if use_concurrent:
        with ThreadPoolExecutor(max_workers=num_proc) as executor:
            for i in range(1, num_proc+1):
                executor.submit(threading_task, i, i, note, gen_num // num_proc, save_path)

    else:
        threads = []
        for i in range(1, num_proc+1):
            thread = threading.Thread(target=threading_task, args=(i, i, note, gen_num // num_proc, save_path))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()
    pass

def stat(args):
    save_path = args.save_path
    with open(save_path, "r", encoding='utf-8') as f:
        # data = json.load(f)
        data = [json.loads(line) for line in f.readlines()]
    
    stat_infos = {
        "num_nodes": [],
        "num_branches": [],
        "num_resistors": [],
        "num_capacitors": [],
        "num_inductors": [],
        "num_voltage_sources": [],
        "num_current_sources": [],
        "num_controlled_sources": [],
        "num_shorts": [],
        "num_voltage_measurements": [],
        "num_current_measurements": [],
    }
    stat_results = {
        "args": vars(args),
        "num_nodes": {},
        "num_branches": {},
        "num_resistors": {},
        "num_capacitors": {},
        "num_inductors": {},
        "num_voltage_sources": {},
        "num_current_sources": {},
        "num_controlled_sources": [],
        "num_shorts": {},
        "num_voltage_measurements": {},
        "num_current_measurements": {},
    }
    for item in data:
        stat_info = item["stat"]
        for key in stat_info:
            stat_infos[key].append(stat_info[key])
    for key in stat_infos:
        print(f"{key}:\n\tmean: {np.mean(stat_infos[key])}\n\tstd: {np.std(stat_infos[key])}\n\tmax: {np.max(stat_infos[key])}\n\tmin: {np.min(stat_infos[key])}\n")
        mmean = float(np.mean(stat_infos[key]))
        sstd = float(np.std(stat_infos[key]))
        mmax = float(np.max(stat_infos[key]))
        mmin = float(np.min(stat_infos[key]))

        stat_results[key] = {
            "mean": mmean,
            "std": sstd,
            "max": mmax,
            "min": mmin
        }
    with open(save_path.replace(".json", "_stat.json"), "w", encoding='utf-8') as f:
        f.write(json.dumps(stat_results, ensure_ascii=False, indent=4))
    

if __name__ == '__main__':
    args = parse_args()
    main(args)
    stat(args)