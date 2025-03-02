import base64
import httpx
import json
import os
#import openai
from openai import OpenAI
# from api_utils import encode_image
import re

from argparse import ArgumentParser
from api_utils import OpenAIImage2TextAPIClient, AnthropicImage2TextAPIClient, GoogleImage2TextAPIClient, ZhipuAIImage2TextAPIClient

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

def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except:
        return None

data_note = 'test_240714'
vlm = "cogagent-vqa-09-02-16-42"
data_root_path = f"./inference/data/{data_note}"
qa_path = os.path.join(data_root_path, "qa_main.json")
cot_data_path = f'./ft_vlm/results/test_{data_note}/{vlm}_eval.json'
output_root_path = "./inference/results"

use_prompt_nosim = True
use_raw_sim_ret = False
extra_note ="nosimprompt" if use_prompt_nosim else ""
extra_note += "_rawsim" if use_raw_sim_ret else ""
print(f"Use prompt nosim: {use_prompt_nosim}, use raw sim ret: {use_raw_sim_ret}")

model_name__ = model_name.replace("-", "")
note = f"{model_name__}_{vlm}_maps_v1_{extra_note}"
note_1 = f"{model_name__}_{vlm}_maps_v1"
output_file_path = os.path.join(output_root_path, data_note, f"{note}_step2.json")
output_file_path_step1 = os.path.join(output_root_path, data_note, f"{note_1}_step1_sim.json")
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

# inference setting 
max_tokens = 2048

prompt_template = "你是一个电路原理专家，你需要求解一道中文电路习题，题目由一张电路图和中文问题组成。我们通过电路识别模型，将电路图转化成了网表语言SPICE代码。同时，我们将SPICE代码放进电路仿真软件中，获得了仿真结果。请你根据电路图、问题、SPICE网表、仿真结果，推理出问题的答案。\n请将最终的数值或者表达式用json格式表示，格式为```json\n{{\n\t待求值1：求解结果1,\n\t待求值2：求解结果2\n}}```。\n如果仿真结果直接包含了答案，你就输出仿真结果得到的答案即可。\n如果仿真出现异常或不包含题目的答案，你需要根据网表和仿真结果进行进一步的推理得到最终结果。\n\n# 问题\n{question}{subq}\n\n # 电路图转成的SPICE网表\n{spice}\n\n # 仿真结果\n{sim_ret}\n\n"

prompt_nosim = "你是一个电路原理专家，你需要求解一道中文电路习题，题目由一张电路图和中文问题组成。我们通过电路识别模型，将电路图转化成了网表语言SPICE代码。请你根据问题、电路图和SPICE网表，推理出问题的答案。\n请将最终的数值或者表达式用json格式表示，格式为```json\n{{\n\t待求值1：求解结果1,\n\t待求值2：求解结果2\n}}```。\n\n# 问题\n{question}{subq}\n\n # 电路图转成的SPICE网表\n{spice}\n\n"

test_subset = None

subquestions = {
    '(a)': '请完成第(a)小题。',
    '(b)': '请完成第(b)小题。',
    '(c)': '请完成第(c)小题。',
    '(d)': '请完成第(d)小题。',
    '(e)': '请完成第(e)小题。',
}

def extract_json_from_content(content):
    try:
        json_ret = re.findall(r'```json(.+?)```', content, re.DOTALL)
        answer = json.loads(json_ret[0])
    except:
        try:
            answer = json.loads(content)
        except:
            answer = None
    return answer

def load_json_or_jsonl(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]   

    return data

def step2_gen_solution(id, question, spice, subquestion, output_file_path, golden_answer=None, sim_ret=""):
    print("#"*50)
    print(f"Processing {id} for step 2...")
    prompt = prompt_template.format(question=question, subq=subquestion, spice=spice, sim_ret=sim_ret)
    if use_prompt_nosim and sim_ret == "No simulation result." or sim_ret == {} or sim_ret == "" or 'error' in sim_ret:
        prompt = prompt_nosim.format(question=question, subq=subquestion, spice=spice)
    
    messages = client.request(prompt, os.path.join(data_root_path, f"{id}.jpg"))
    print(messages)

    res = {}
    res['id'] = id
    res['question'] = question
    res['prompt'] = prompt
    res['golden_answer'] = golden_answer
    
    try:
        response_content = client.parse(messages)
        res['gen_solution'] = response_content
        answer = extract_json_from_content(response_content)
        res['answer'] = answer
    except:
        print("Error in inference.")
        return

    with open(output_file_path, 'a', encoding='utf-8') as json_file:
        json_file.write(json.dumps(res, ensure_ascii=False) + '\n')

def main():
    done_ids = []
    step1_data = []
    if os.path.exists(output_file_path):
        with open(output_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                done_ids.append(item['id'])
    print(f"Load done ids from {output_file_path}, got {len(done_ids)} items.")
    
    if os.path.exists(output_file_path_step1):
        with open(output_file_path_step1, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                step1_data.append(item)
    print(f"Load Step 1 data from {output_file_path_step1}, got {len(step1_data)} items.")
    
    qa_data = load_json_or_jsonl(qa_path)
    cot_data = load_json_or_jsonl(cot_data_path)
    
    id_key_cot = 'id' if 'id' in cot_data[0] else 'qid'
    id2cotitem = {item[id_key_cot]: item for item in cot_data}
    id2step1item = {item['id']: item for item in step1_data}
    
    results = []    
    for item in qa_data:
        id = item['id'] if 'id' in item else item['qid']
        if item['is_forward'] == False:
            print(f"Skip inference for {id}, is_forward is False.")
            continue

        try:
            spice = id2step1item[id]['spice']
            sim_ret_noraw = id2step1item[id]['sim_ret'].copy()
            sim_ret_noraw.pop('raw_file', None)
            sim_ret = id2step1item[id]['sim_ret']['raw_file'] if use_raw_sim_ret else sim_ret_noraw
        except Exception as e:
            print(f"error load spice or sim_ret from step1 for {id}: {e}")
            continue

        if test_subset and id not in test_subset:
            continue

        question = item['question']
        for subq in subquestions:
            if subq in id:
                subquestion = subquestions[subq]
                break
        else:
            subquestion = ''

        if id in done_ids:
            print(f"Skip inference for {id}, id in done_ids.")
            continue

        step2_gen_solution(id, question=question, spice=spice, subquestion=subquestion, output_file_path=output_file_path, golden_answer=item['golden_answer'], sim_ret=sim_ret)

if __name__ == '__main__':
    main()