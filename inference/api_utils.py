from openai import OpenAI
from zhipuai import ZhipuAI
import anthropic
import google.generativeai as genai
import json
from argparse import ArgumentParser
from multiprocessing import Pool, Manager
from tqdm import tqdm
import time
import os
import httpx
import base64
import PIL

# Replace with your actual API key
client = None
proxy_url = "http://127.0.0.1:7890"
httpx_client = httpx.Client(proxies={"http://": proxy_url, "https://": proxy_url})
max_tokens = 2048

def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except:
        return None
    
class Image2TextAPIClient:
    def __init__(self, api_key, base_url, model_name, system_prompt=''):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.system_prompt = system_prompt

    def request(self, user_prompt, image_path, **kwargs):
        raise NotImplementedError

    def parse(self, response):
        raise NotImplementedError
    
class OpenAIImage2TextAPIClient(Image2TextAPIClient):

    def __init__(self, api_key, model_name, system_prompt='', base_url=None):
        super().__init__(api_key, base_url, model_name, system_prompt)
        self.client = OpenAI(api_key=api_key, 
                             base_url=base_url,
                             http_client=httpx_client
                             )
    
    def request(self, user_prompt, image_path, **kwargs):
        content=[]
        content.append({
                        "type": "text",
                        "text": user_prompt
                })
        image_data = encode_image(image_path)
        if image_data:
            # content.append(
            #     {
            #         "type": "text",
            #         "text": "图{}".format(id),
            #     }
            # )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    }
                }
            )
        messages = self.client.chat.completions.create(
            model=self.model_name,
            max_tokens=max_tokens,
            messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        )
        print(messages)
        # response = messages.choices[0].message.content
        return messages
    
    def parse(self, response):
        return response.choices[0].message.content
    
class O1Image2TextAPIClient(OpenAIImage2TextAPIClient):

    def __init__(self, api_key, model_name, system_prompt='', base_url=None):
        super().__init__(api_key, model_name, system_prompt)

    def request(self, user_prompt, image_path, **kwargs):
        content=[]
        content.append({
                        "type": "text",
                        "text": user_prompt
                })
        image_data = encode_image(image_path)
        if image_data:
            # content.append(
            #     {
            #         "type": "text",
            #         "text": "图{}".format(id),
            #     }
            # )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    }
                }
            )
        messages = self.client.chat.completions.create(
            model=self.model_name,
            max_completion_tokens=max_tokens,
            messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        )
        print(messages)
        # response = messages.choices[0].message.content
        return messages
    def parse(self, response):
        return super().parse(response)
    
class AnthropicImage2TextAPIClient(Image2TextAPIClient):

    def __init__(self, api_key, model_name, system_prompt='', base_url=None):
        super().__init__(api_key, base_url, model_name, system_prompt)
        self.client = anthropic.Anthropic(api_key=api_key, 
                                          http_client=httpx_client
                                          )
    
    def request(self, user_prompt, image_path=None):
        content=[]
        content.append({
                        "type": "text",
                        "text": user_prompt
                })
        if not image_path:
            image_data = None
        else:
            image_data = encode_image(image_path)
            if image_path.endswith('.png'):
                image_media_type = "image/png"
            elif image_path.endswith('.jpg'):
                image_media_type = "image/jpeg"
            else:
                raise ValueError("Unsupported image format")
        
        if image_data:
            # content.append(
            #     {
            #         "type": "text",
            #         "text": "图{}".format(id),
            #     }
            # )
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_media_type,
                        "data": image_data,
                    },
                },
            )
        messages = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        )

        print(messages)
        # response = messages.choices[0].message.content
        return messages
    
    def parse(self, response):
        return response.content[0].text
    
class GoogleImage2TextAPIClient(Image2TextAPIClient):
    def __init__(self, api_key,  model_name, system_prompt='',base_url=None):
        super().__init__(api_key, base_url, model_name, system_prompt)
        os.environ["HTTP_PROXY"] = proxy_url
        os.environ["HTTPS_PROXY"] = proxy_url
        genai.configure(api_key=api_key, transport="rest")

        self.client = genai.GenerativeModel(model_name=model_name)

    def request(self, user_prompt, image_path=None):
        if image_path:
            sample_file = PIL.Image.open(image_path)
        # Prompt the model with text and the previously uploaded image.
            response = self.client.generate_content([sample_file, user_prompt]) 
        else:
            response = self.client.generate_content([user_prompt])
        return response

    def parse(self, response):
        return response.text

class ZhipuAIImage2TextAPIClient(Image2TextAPIClient):
    def __init__(self, api_key, model_name, system_prompt='', base_url=None):
        super().__init__(api_key, base_url, model_name, system_prompt)
        self.client = ZhipuAI(api_key="84d5e83e57189d811651b114e67ffff2.0KIiVaU1YT1mOG6E", 
                              )

    def request(self, user_prompt, image_path=None):
        image_data = encode_image(image_path)
        if image_data:
            image_media_type = "image/jpeg"
            messages = self.client.chat.completions.create(
                model=self.model_name,
                max_tokens=max_tokens,
                messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data
                            }
                        },
                    ],
                }
            ],
            )
        else:
            # raise ValueError("Unsupported image format")
            messages = self.client.chat.completions.create(
                model=self.model_name,
                max_tokens=max_tokens,
                messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt,
                        },
                    ],
                }
            ],
            )
        print(messages)
        return messages

    def parse(self, response):
        return response.choices[0].message.content