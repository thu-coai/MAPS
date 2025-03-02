import base64
import httpx
import json
import os
#import openai
from openai import OpenAI
from api_utils import encode_image, OpenAIImage2TextAPIClient, ZhipuAIImage2TextAPIClient, GoogleImage2TextAPIClient, AnthropicImage2TextAPIClient
import re

from argparse import ArgumentParser

proxy_url = "http://127.0.0.1:7890"
httpx_client = httpx.Client(proxies={"http://": proxy_url, "https://": proxy_url})

parser = ArgumentParser()

parser.add_argument("--model_name", type=str, default="gpt-4-vision-preview")
args = parser.parse_args()

model_name = args.model_name

API_KEY_GPT = os.getenv("API_KEY_GPT")
API_KEY_GLM = os.getenv("API_KEY_GLM")
API_KEY_CLAUDE = os.getenv("API_KEY_CLAUDE")
API_KEY_GEMINI = os.getenv("API_KEY_GEMINI")

client = None
if 'gpt' in model_name:
    client = OpenAIImage2TextAPIClient(
        api_key=API_KEY_GPT,
        model_name=model_name
    )
elif 'claude' in model_name:
    client = AnthropicImage2TextAPIClient(
        api_key=API_KEY_CLAUDE,
        model_name=model_name,
    )
elif 'gemini' in model_name:
    client = GoogleImage2TextAPIClient(
        api_key=API_KEY_GEMINI,
        model_name=model_name
    )
elif 'glm' in model_name:
    client = ZhipuAIImage2TextAPIClient(
        api_key=API_KEY_GLM,
        model_name=model_name
    )
else:
    raise ValueError(f"Invalid model name: {model_name}") 
print(f"Using model: {model_name}, Client: {client}")


data_note = 'simple_circuit_eval'
vlm = 'cogagent-vqa-09-02-16-42'
data_root_path = f"inference/data/{data_note}"
qa_path = os.path.join(data_root_path, "qa.json")
cot_data_path = f'ft_vlm/results/test_{data_note}/{vlm}_eval.json'
output_root_path = "inference/results"

model_name__ = model_name.replace("-", "")
note = f"{model_name__}_{vlm}_maps_v1"
output_file_path = os.path.join(output_root_path, data_note, f"{note}_step1.json")
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

# inference setting 
max_tokens = 2048

prompt_template = "你是一个电路原理专家，你是一个电路原理的专家，你需要求解一道中文电路习题，题目由一张电路图和中文问题组成。我们通过电路识别模型，将电路图转化成了网表语言SPICE代码。但是，电路图的网表中可能遗漏了一些数值（标为<Empty>），你需要根据题干信息，补全SPICE代码，例如将题目中的电阻的值填入网表中。你的输出为完善后的SPICE代码，请将答案放在```SPICE 和 ```之间。\n\n# 问题\n{question}{subq}\n\n # 电路图转成的SPICE网表\n{spice}\n\n"
use_image = False
test_subset = None
# test_subset = ['1-14(a)']

subquestions = {
    '(a)': '请完成第(a)小题。',
    '(b)': '请完成第(b)小题。',
    '(c)': '请完成第(c)小题。',
    '(d)': '请完成第(d)小题。',
    '(e)': '请完成第(e)小题。',
}

def load_json_or_jsonl(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]   

    return data

def step1_refine_spice(id, question, pred_spice, subquestion, output_file_path):
    print("#"*50)
    print(f"Processing {id} for step 1...")
    prompt = prompt_template.format(question=question, subq=subquestion, spice=pred_spice)
    print(prompt)

    res = {}
    res['id'] = id
    res['question'] = question
    res['prompt'] = prompt
    
    if '<empty>' in pred_spice.lower():
        # try:
        # Get SPICE refinement
        messages = client.request(prompt, image_path=None)
        response_content = client.parse(messages)

        res['gen_solution'] = response_content

        try: 
            spice = re.search(r'```SPICE(.*?)```', response_content, re.S).group(1)
            res['spice'] = spice
            res['spice_refined'] = True

            print(f"Refined SPICE: {spice}")
        # results.append(res)
        except Exception as e:
            print(f"Error in extract SPICE: {e}")
            res['spice'] = pred_spice
            res['spice_refined'] = False
    else:
        print("No need to refine.")
        res['spice'] = pred_spice
        res['spice_refined'] = False

    with open(output_file_path, 'a', encoding='utf-8') as json_file:
        json_file.write(json.dumps(res, ensure_ascii=False) + '\n')

def main():
    done_ids = []
    if os.path.exists(output_file_path):
        with open(output_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                done_ids.append(item['id'])
    
    qa_data = load_json_or_jsonl(qa_path)
    cot_data = load_json_or_jsonl(cot_data_path)
    id_key_cotdata = 'id' if 'id' in cot_data[0] else 'qid'
    id2spice = {item[id_key_cotdata]: item['pred'] for item in cot_data}
    
    results = []    
    for item in qa_data:
        id = item['id'] if 'id' in item else item['qid']
        question = item['question']
        spice = id2spice[id]

        for subq in subquestions:
            if subq in id:
                subquestion = subquestions[subq]
                break
        else:
            subquestion = ''

        if id in done_ids:
            print(f"Skip inference for {id}... which has been done.")
            continue
        if item["is_forward"] == False:
            print(f"Skip inference for {id}... which is not forward.")
            continue
        step1_refine_spice(id, question=question, pred_spice=spice, subquestion=subquestion, output_file_path=output_file_path)

if __name__ == '__main__':
    main()