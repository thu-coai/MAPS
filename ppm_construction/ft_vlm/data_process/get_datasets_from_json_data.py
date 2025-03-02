import os

import json
import concurrent.futures
import subprocess

from utils.dataprocess_utils import pdf2jpg, compile_latex, preprocess_latex

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--note", type=str, required=True)
args = parser.parse_args()

label_key = "spice"

note = args.note
file_path = f"data_syn/data/{note}.json"
dataset_path = "datasets/{}".format(note)
img_suffix = ".jpg"

def complie_latex_codes(latex_codes, output_dir, recompile=True):
    """
    Compile a list of latex codes into pdf files.
    latex_code: {
        "id": "xxx",
        "latex": "xxx",
        ...
    }
    """

    def compile_latex_code(latex_, recompile=recompile):
        print(f"compiling {latex_['id']}...")
        # file_name = os.path.basename(latex_file).split(".")[0]

        file_name = latex_['id']

        if not recompile and os.path.exists(f"{output_dir}/{file_name}.pdf"):
            print(f"{output_dir}/{file_name}.pdf already exists, skip compiling...")
            return True

        latex_code = latex_['latex']

        processed_code = preprocess_latex(latex_code)
        print(f"compile {file_name}.tex to {output_dir}/{file_name}.pdf...")
        succ = compile_latex(output_dir, file_name, processed_code)

        if succ:
            os.system(f"rm {output_dir}/{file_name}.aux")
            os.system(f"rm {output_dir}/{file_name}.log")

        return succ
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(compile_latex_code, latex_codes)

    compiled_codes, not_compiled_latex_codes = check_compiled_latex_codes(latex_codes, output_dir=output_dir)
    print(f"not_compiled_latex_codes: {not_compiled_latex_codes}")
    
    # os.system(f"rm {output_dir}/**.aux")
    # os.system(f"rm {output_dir}/**.log")
    return list(results), compiled_codes, not_compiled_latex_codes


def make_datasets(latex_codes, dataset_path, zoom_x=1, zoom_y=1, rotation_angle=0, save=True, rm_suffixs=[".tex", ".pdf"]):
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
    labels = {}
    num_ai, num_grid, num_other = 0, 0, 0
    num_ai_valid, num_grid_valid, num_other_valid = 0, 0, 0

    for latex_ in latex_codes:
        img_key = latex_['id']
        pdf_path = os.path.join(dataset_path, img_key + ".pdf")
        img_path = os.path.join(dataset_path, img_key + img_suffix)

        label_content = latex_[label_key]
        
        labels[img_key] = label_content

        if "gpt" in img_key:
            num_ai += 1
        elif "grid" in img_key:
            num_grid += 1
        else:
            num_other += 1

        if save:
            succ = pdf2jpg(pdfPath=pdf_path, imgPath=img_path, zoom_x=zoom_x, zoom_y=zoom_y, rotation_angle=rotation_angle)

            if succ:
                if "gpt" in img_key:
                    num_ai_valid += 1
                elif "grid" in img_key:
                    num_grid_valid += 1
                else:
                    num_other_valid += 1

        for suff in rm_suffixs:
            os.system(f"rm {dataset_path}/{img_key}{suff}")
        
    if save:
        with open(os.path.join(dataset_path, "labels.json"), "w") as f:
            json.dump(labels, f, indent=4)
    print(f"num_ai: {num_ai}, num_grid: {num_grid}, num_other: {num_other}")
    print(f"num_ai_valid: {num_ai_valid}, num_grid_valid: {num_grid_valid}, num_other_valid: {num_other_valid}")

def check_compiled_latex_codes(latex_codes, output_dir):
    """
    Check if the pdf files corresponding to the latex files are already compiled.
    """
    def check_compiled(latex_code):
        # file_name = os.path.basename(latex_file).split(".")[0]
        file_name = latex_code['id']
        return {file_name: os.path.exists(f"{output_dir}/{file_name}.pdf")}
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(check_compiled, latex_codes)
    for result in results:
        print(result)
    not_compiled_files = [file for file in results if not list(file.values())[0]]
    compiled_files = [file for file in results if list(file.values())[0]]
    return compiled_files, not_compiled_files

def debug():
    pass

def main():

    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(l) for l in f]
    
    print(f"Got {len(data)} data items ...")

    _, done, not_done = complie_latex_codes(data, output_dir=dataset_path, recompile=False)

    print(f"Compiled {len(done)} files, {len(not_done)} files not compiled ...")

    make_datasets(data, dataset_path, zoom_x=1, zoom_y=1, rotation_angle=0, rm_suffixs=[])

if __name__ == "__main__":
    # debug()
    main()
