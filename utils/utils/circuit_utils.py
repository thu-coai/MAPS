import os
import logging
import random
import logging
import jsonlines
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
from sat.helpers import print_rank0
import json

from datetime import datetime
import numpy as np

def compile_latex(folder, file_name, latex_code):
    succ = False
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(f"{folder}/{file_name}.tex", "w") as f:
        f.write(latex_code)
    try:
        exit_code = os.system(f"pdflatex -interaction=batchmode -output-directory={folder} {folder}/{file_name}.tex")
        if exit_code == 0 and os.path.exists(f"{folder}/{file_name}.pdf"):
            print("Successfully compiled!")
            succ = True
        else:
            exit_code = os.system(f"xelatex -interaction=batchmode -output-directory={folder} {folder}/{file_name}.tex")
            if exit_code == 0 and os.path.exists(f"{folder}/{file_name}.pdf"):
                print("Successfully compiled!")
                succ = True
            else:
                print("Failed to compile.")
                # delete failed
    except Exception as e:
        print(e)

    return succ

def find_all_files(path, suffix=".jpg", cur_dir=False):
    target_files = []
    if cur_dir:
        items = os.listdir(path)
        target_files = [item for item in items 
                        if os.path.isfile(os.path.join(path, item)) and item.endswith(suffix)]
    else:
        for cur_dir, _, files in os.walk(path, followlinks=True):
            for f in files:
                if f.endswith(suffix):
                    target_files.append(os.path.join(cur_dir, f))
            
    print_rank0(f'find {len(target_files)} files...')
    print(f'target_files: {target_files}')
    return target_files

class ImageLabelsDataset(Dataset):
    def __init__(self, image_processor, text_processor, args, data_dirs, cross_image_processor=None, prompt="", img_suffix='.jpg', **kwargs):
        super().__init__()
        self.prompt = prompt
        self.image_processor, self.text_processor, self.cross_image_processor = image_processor, text_processor, cross_image_processor
        self.img_suffix = img_suffix
        self.get_image_from_cur_dir = args.get_image_from_cur_dir
        self.data, self.labels = self.load_data(data_dirs)

    def process_img(self, img):
        img_dict = {'vision': self.image_processor(img)}
        if self.cross_image_processor:
            img_dict.update({'cross': self.cross_image_processor(img)})
        return img_dict
    
    def process_text(self, answer, prompt):
        return self.text_processor(answer, prompt)
    
    def load_data(self, data_dir):
        all_files = find_all_files(data_dir, suffix=self.img_suffix, cur_dir=self.get_image_from_cur_dir)
        with open(f"{data_dir}/labels.json", "r") as f:
            labels = json.load(f)
        print_rank0(f"find {(all_files)} samples in all...")
        return all_files, labels
    
    def get_key(self, path):
        return path.split('/')[-1].replace(self.img_suffix, "")
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        # img
        try:
            img = Image.open(data).convert('RGB')
        except Exception as e:
            print_rank0(e, level=logging.WARNING)
            return {}
        img_dict = self.process_img(img)
        # text
        # label = data.split('/')[-1].split('.')[0]
        # uni_key = label
        uni_key = self.get_key(data)
        assert uni_key in self.labels, f"{uni_key} not in labels, {self.labels.keys()}"
        label = self.labels[uni_key]

        text_dict = self.process_text(label, self.prompt)
        if text_dict is None:
            print_rank0(f"Process text failed. Please check the max_target_length & max_source_length.\n The data is {data}", level=logging.WARNING)
            return {}
        # other attr
        ret = {**img_dict, **text_dict, "question_id": uni_key}
        # print(ret)
        # exit()
        return ret

class ItemDataset(Dataset):
    def __init__(self, image_processor, text_processor, args, data_dirs, cross_image_processor=None, **kwargs):
        super().__init__()
        self.data = self.load_data(data_dirs)
        self.image_processor, self.text_processor, self.cross_image_processor = image_processor, text_processor, cross_image_processor
    
    def process_img(self, img):
        img_dict = {'vision': self.image_processor(img)}
        if self.cross_image_processor:
            img_dict.update({'cross': self.cross_image_processor(img)})
        return img_dict
    
    def process_text(self, answer, prompt):
        return self.text_processor(answer, prompt)
    
    def load_data(self, data_dir):
        all_files = find_all_files(data_dir, suffix=".jpg")
        print_rank0(f"find {len(all_files)} samples in all...")
        return all_files
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        # img
        try:
            img = Image.open(data).convert('RGB')
        except Exception as e:
            print_rank0(e, level=logging.WARNING)
            return {}
        img_dict = self.process_img(img)
        # text
        label = data.split('/')[-1].split('.')[0]
        uni_key = label
        text_dict = self.process_text(label, "CAPTCHA:")
        if text_dict is None:
            print_rank0(f"Process text failed. Please check the max_target_length & max_source_length.\n The data is {data}", level=logging.WARNING)
            return {}
        # other attr
        ret = {**img_dict, **text_dict, "question_id": uni_key}
        return ret