import os
import shutil
import json

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--note", type=str, default="grid_v10_240812")
args = parser.parse_args()

dataset_name = args.note

raw_dataset_path = f"datasets/{dataset_name}"
splited_dataset_path = f"datasets/{dataset_name}_split"

splits = ['train', 'valid', 'dev', 'test']
# split_ratio = [0.24, 0.02, 0.04]
split_ratio = [0.8, 0.09, 0.01, 0.1]
seed = 2023

def find_all_files(path, suffix=".jpg"):
    target_files = []
    for cur_dir, _, files in os.walk(path, followlinks=True):
        for f in files:
            if f.endswith(suffix):
                target_files.append(os.path.join(cur_dir, f))
    print(f'find {len(target_files)} files...')
    return target_files

all_files = find_all_files(raw_dataset_path)
with open(f"{raw_dataset_path}/labels.json", "r") as f:
    labels = json.load(f)

# split
os.makedirs(f"{splited_dataset_path}", exist_ok=True)
for split in splits:
    os.makedirs(f"{splited_dataset_path}/{split}", exist_ok=True)

import random
random.seed(seed)
random.shuffle(all_files)
num_datasets = len(all_files)
print(f"num_datasets: {num_datasets} num_use: {int(num_datasets*sum(split_ratio))}")


split2files = {
}
for i in range(len(splits)):
    ratio = split_ratio[i]
    split = splits[i]
    start_idx = int(sum(split_ratio[:i])*num_datasets) if i != 0 else 0
    split_data = all_files[start_idx:start_idx+int(ratio*num_datasets)]
    split2files[split] = split_data


# def get_key(path):
#     return path.split("/")[-1].replace(".jpg", "")

def get_path(key):
    return os.path.join(raw_dataset_path, key+".jpg")

# train_labels = {k:v for k,v in labels.items() if get_path(k) in train}
# valid_labels = {k:v for k,v in labels.items() if get_path(k) in valid}
# test_labels = {k:v for k,v in labels.items() if get_path(k) in test}

split2labels = {}
for split in splits:
    split2labels[split] = {k:v for k,v in labels.items() if get_path(k) in split2files[split]}

# check if valid and test labels are in train
valid_labels = split2labels["valid"] if "valid" in split2labels else {}
test_labels = split2labels["test"] if "test" in split2labels else {}
train_labels = split2labels["train"]
valid_in_train_keys = [key for key in valid_labels.values() if key in train_labels.values()]
test_in_train_keys = [key for key in test_labels.values() if key in train_labels.values()]
num_valid_in_train, num_test_in_train = len(valid_in_train_keys), len(test_in_train_keys)
print(f"num_valid_in_train: {num_valid_in_train}, num_test_in_train: {num_test_in_train}")

# print(f"num_train: {len(train)}, num_valid: {len(valid)}, num_test: {len(test)}")
for split in splits:
    print(f"num_{split}: {len(split2files[split])}, num_{split}_labels: {len(split2labels[split])}\n\n")

def regulize_path(path):
    key = path.replace(" ","").replace("_（", "").replace("）", "").replace("（", "_")
    return key

for split in splits:
    print(f"building {split}")
    cnt = 0
    for file in split2files[split]:
        new_file_name = regulize_path(file.split("/")[-1])
        try:
            assert new_file_name.replace('.jpg', '') in split2labels[split], f"{new_file_name.replace('.jpg', '')} not in {split}_labels"
        except:
            print(f"{new_file_name.replace('.jpg', '')} not in {split}_labels")
            continue
        shutil.copy(file, os.path.join(f"{splited_dataset_path}/{split}", new_file_name))
        cnt += 1
    print(f"{split} done, {cnt} files")
    
    with open(os.path.join(f"{splited_dataset_path}/{split}", "labels.json"), "w") as f:
        # json.dump(eval(f"{split}_labels"), f, indent=4)
        json.dump(split2labels[split], f, indent=4)
    print(f"building {split} done")

# print("building train")
# for file in train:
#     shutil.copy(file, os.path.join(f"{splited_dataset_path}/train", file.split("/")[-1]))
# print("building valid")
# for file in valid:
#     shutil.copy(file, os.path.join(f"{splited_dataset_path}/train", file.split("/")[-1]))
# print("building test")
# for file in test:
#     shutil.move(file, os.path.join("archive_split/test", file.split("/")[-1]))
print("done")