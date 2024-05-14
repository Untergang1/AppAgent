import re
from abc import abstractmethod
from typing import List, Optional
from http import HTTPStatus

import requests
import dashscope
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, BitsAndBytesConfig
from PIL import Image

from .utils import print_with_color, encode_image


class BaseModel:
    def __init__(self):
        pass


class BaseLanguageModel(BaseModel):
    def __int__(self):
        pass

    @abstractmethod
    def get_model_response(self, prompt: str) -> (bool, str):
        pass


class BaseMultiModalModel(BaseModel):
    def __int__(self):
        pass

    @abstractmethod
    def get_model_response(self, prompt: str, images: List[str]) -> (bool, str):
        pass


class OpenAIModel(BaseMultiModalModel):
    def __init__(self, base_url: str, api_key: str, model: str, temperature: float, max_tokens: int):
        super().__init__()
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def get_model_response(self, prompt: str, images: List[str]) -> (bool, str):
        content = [
            {
                "type": "text",
                "text": prompt
            }
        ]
        for img in images:
            base64_img = encode_image(img)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_img}"
                }
            })
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        response = requests.post(self.base_url, headers=headers, json=payload).json()
        if "error" not in response:
            usage = response["usage"]
            prompt_tokens = usage["prompt_tokens"]
            completion_tokens = usage["completion_tokens"]
            print_with_color(f"Request cost is "
                             f"${'{0:.2f}'.format(prompt_tokens / 1000 * 0.01 + completion_tokens / 1000 * 0.03)}",
                             "yellow")
        else:
            return False, response["error"]["message"]
        return True, response["choices"][0]["message"]["content"]


class QwenModel(BaseMultiModalModel):
    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        dashscope.api_key = api_key

    def get_model_response(self, prompt: str, images: List[str]) -> (bool, str):
        content = [{
            "text": prompt
        }]
        for img in images:
            img_path = f"file://{img}"
            content.append({
                "image": img_path
            })
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        call_args = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_k": 1
        }
        response = dashscope.MultiModalConversation.call(
            model=self.model,
            messages=messages,
            **call_args
        )
        if response.status_code == HTTPStatus.OK:
            return True, response.output.choices[0].message.content[0]["text"]
        else:
            return False, response.message

class QwenTextModel(BaseLanguageModel):
    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        dashscope.api_key = api_key

    def get_model_response(self, prompt: str) -> (bool, str):
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        call_args = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_k": 1
        }
        response = dashscope.Generation.call(
            model=self.model,
            messages=messages,
            **call_args
        )
        if response.status_code == HTTPStatus.OK:
            return True, response.output.text
        else:
            return False, response.message

class GeminiModel(BaseMultiModalModel):
    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = f"https://generativelanguage.googleapis.com/v1/models/{self.model}:generateContent?key={self.api_key}"

    def get_model_response(self, prompt: str, images: List[str]) -> (bool, str):
        parts=[
            {"text": prompt}
        ]
        for img in images:
            base64_img = encode_image(img)
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": base64_img
                    }
            })
        headers = {
            "Content-Type": "application/json",
        }
        payload = {
            "contents": [{"parts":parts}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens
            }
        }

        response = requests.post(self.base_url, headers=headers, json=payload).json()
        if "error" not in response:
            return True, response['candidates'][0]['content']['parts'][0]['text']
        else:
            return False, response["error"]

class GeminiTextModel(BaseLanguageModel):
    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = f"https://generativelanguage.googleapis.com/v1/models/{self.model}:generateContent?key={self.api_key}"

    def get_model_response(self, prompt: str) -> (bool, str):
        parts=[
            {"text": prompt}
        ]
        headers = {
            "Content-Type": "application/json",
        }
        payload = {
            "contents": [{"parts":parts}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens
            }
        }

        response = requests.post(self.base_url, headers=headers, json=payload).json()
        if "error" not in response:
            return True, response['candidates'][0]['content']['parts'][0]['text']
        else:
            return False, response["error"]

class CogVLM(BaseModel):
    def __init__(self, temperature: float = 0, max_tokens: int = 500):
        super().__init__()
        torch.set_default_device("cuda")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForCausalLM.from_pretrained(
            "THUDM/cogagent-chat-hf",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            quantization_config=bnb_config,
            trust_remote_code=True
        ).eval()
        tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5", trust_remote_code=True)
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.torch_type = torch.float16

    def get_model_response(self, prompt: str, images: List[str]) -> (bool, str):
        prompt_template="A chat between a curious user and an artificial intelligence assistant.USER:{prompt} ASSISTANT:"
        prompt = prompt_template.format(prompt=prompt)
        # print(prompt)
        images = [Image.open(image).convert('RGB') for image in images]
        input_by_model = self.model.build_conversation_input_ids(self.tokenizer, query=prompt, images=images, template_version='base')

        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0),
            'images': [[input_by_model['images'][0].to(self.torch_type)]],
            'cross_images': [[input_by_model['cross_images'][0].to(self.torch_type)]]
        }
        gen_kwargs = {"max_new_tokens": self.max_tokens,
                      "do_sample": False}
        outputs = self.model.generate(**inputs, **gen_kwargs)[0]
        outputs = outputs[inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(outputs)
        message = response.split("</s>")[0]
        return True, message


class IMPModel(BaseModel):
    def __init__(self, temperature: float = 0, max_tokens: int = 500):
        super().__init__()
        torch.set_default_device("cuda")

        model = AutoModelForCausalLM.from_pretrained(
            "MILVLG/imp-v1-3b",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained("MILVLG/imp-v1-3b", trust_remote_code=True)
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_tokens = max_tokens

    def get_model_response(self, prompt: str, images: List[str]) -> (bool, str):
        prompt_template="A chat between a curious user and an artificial intelligence assistant.USER:{prompt} ASSISTANT:"
        prompt = len(images)*"\nimage:<image>" + prompt
        prompt = prompt_template.format(prompt=prompt)
        input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids
        images=[Image.open(image) for image in images]
        image_tensors = [self.model.image_preprocess(image) for image in images]

        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=self.max_tokens,
            images=image_tensors,
            use_cache=True)[0]
        message = self.tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
        return True, message

def chose_model(model,configs):
    mllm=None
    if model == "OpenAI":
        mllm = OpenAIModel(base_url=configs["OPENAI_API_BASE"],
                           api_key=configs["OPENAI_API_KEY"],
                           model=configs["OPENAI_API_MODEL"],
                           temperature=configs["TEMPERATURE"],
                           max_tokens=configs["MAX_TOKENS"])
    elif model == "Qwen":
        mllm = QwenModel(api_key=configs["DASHSCOPE_API_KEY"],
                         model=configs["QWEN_MODEL"],
                         temperature=configs["TEMPERATURE"],
                         max_tokens=configs["MAX_TOKENS"],)
    elif model == "Qwen-text":
        mllm = QwenTextModel(api_key=configs["DASHSCOPE_API_KEY"],
                         model=configs["QWEN_TEXT_MODEL"],
                         temperature=configs["TEMPERATURE"],
                         max_tokens=configs["MAX_TOKENS"],)
    elif model == "IMP":
        mllm = IMPModel()
    elif model == "Gemini":
        mllm = GeminiModel(api_key=configs["GEMINI_API_KEY"],
                           model=configs["GEMINI_MODEL"],
                           temperature=configs["TEMPERATURE"],
                           max_tokens=configs["MAX_TOKENS"])
    elif model == "Gemini-text":
        mllm = GeminiTextModel(api_key=configs["GEMINI_API_KEY"],
                           model=configs["GEMINI_TEXT_MODEL"],
                           temperature=configs["TEMPERATURE"],
                           max_tokens=configs["MAX_TOKENS"])
    elif model == "CogVLM":
        mllm = CogVLM()
    return mllm


def parse_explore_rsp(rsp, detail=True):
    try:
        observation = re.findall(r"Observation: (.*?)$", rsp, re.MULTILINE)[0]
        think = re.findall(r"Thought: (.*?)$", rsp, re.MULTILINE)[0]
        act = re.findall(r"Action: (.*?)$", rsp, re.MULTILINE)[0]
        last_act = re.findall(r"Summary: (.*?)$", rsp, re.MULTILINE)[0]
        if detail:
            print_with_color("Observation:", "yellow")
            print_with_color(observation, "magenta")
            print_with_color("Thought:", "yellow")
            print_with_color(think, "magenta")
            print_with_color("Action:", "yellow")
            print_with_color(act, "magenta")
            print_with_color("Summary:", "yellow")
            print_with_color(last_act, "magenta")
        if "FINISH" in act:
            return ["FINISH"]
        act_name = act.split("(")[0]
        if act_name == "tap":
            area = int(re.findall(r"tap\((.*?)\)", act)[0])
            return [act_name, area, last_act]
        elif act_name == "text":
            input_str = re.findall(r"text\((.*?)\)", act)[0][1:-1]
            return [act_name, input_str, last_act]
        elif act_name == "long_press":
            area = int(re.findall(r"long_press\((.*?)\)", act)[0])
            return [act_name, area, last_act]
        elif act_name == "swipe":
            params = re.findall(r"swipe\((.*?)\)", act)[0]
            area, swipe_dir, dist = params.split(",")
            area = int(area)
            swipe_dir = swipe_dir.strip()[1:-1]
            dist = dist.strip()[1:-1]
            return [act_name, area, swipe_dir, dist, last_act]
        elif act_name == "grid":
            return [act_name]
        else:
            print_with_color(f"ERROR: Undefined act {act_name}!", "red")
            return ["ERROR"]
    except Exception as e:
        print_with_color(f"ERROR: an exception occurs while parsing the model response: {e}", "red")
        print_with_color(rsp, "red")
        return ["ERROR"]

def parse_explore_rsp_text(rsp, detail=False):
    try:
        observation = re.findall(r"Observation: (.*?)$", rsp, re.MULTILINE)[0]
        think = re.findall(r"Thought: (.*?)$", rsp, re.MULTILINE)[0]
        act = re.findall(r"Action: (.*?)$", rsp, re.MULTILINE)[0]
        last_act = re.findall(r"Summary: (.*?)$", rsp, re.MULTILINE)[0]
        if detail:
            print_with_color("Observation:", "yellow")
            print_with_color(observation, "magenta")
            print_with_color("Thought:", "yellow")
            print_with_color(think, "magenta")
            print_with_color("Action:", "yellow")
            print_with_color(act, "magenta")
            print_with_color("Summary:", "yellow")
            print_with_color(last_act, "magenta")
        if "Stop" in act:
            return ["Stop"]
        act_name = act.split("(")[0]
        if act_name == "Click":
            bounds = re.findall(r"Click\((.*?)\)", act)[0]
            matchs = re.match(r"\[(\w+), (\w+)]\[(\w+), (\w+)]", bounds)
            x1, y1, x2, y2 = matchs.groups()
            return [act_name, ((x1, y1), (x2, y2)), last_act]
        elif act_name == "Type":
            input_str = re.findall(r"text\((.*?)\)", act)[0][1:-1]
            return [act_name, input_str, last_act]
        elif act_name == "Swipe":
            direction = re.findall(r"swipe\((.*?)\)", act)[0]
            return [act_name, direction, last_act]
        else:
            print_with_color(f"ERROR: Undefined act {act_name}!", "red")
            return ["ERROR"]
    except Exception as e:
        print_with_color(f"ERROR: an exception occurs while parsing the model response: {e}", "red")
        print_with_color(rsp, "red")
        return ["ERROR"]

def parse_grid_rsp(rsp, detail=False):
    try:
        observation = re.findall(r"Observation: (.*?)$", rsp, re.MULTILINE)[0]
        think = re.findall(r"Thought: (.*?)$", rsp, re.MULTILINE)[0]
        act = re.findall(r"Action: (.*?)$", rsp, re.MULTILINE)[0]
        last_act = re.findall(r"Summary: (.*?)$", rsp, re.MULTILINE)[0]
        if detail:
            print_with_color("Observation:", "yellow")
            print_with_color(observation, "magenta")
            print_with_color("Thought:", "yellow")
            print_with_color(think, "magenta")
            print_with_color("Action:", "yellow")
            print_with_color(act, "magenta")
            print_with_color("Summary:", "yellow")
            print_with_color(last_act, "magenta")
        if "FINISH" in act:
            return ["FINISH"]
        act_name = act.split("(")[0]
        if act_name == "tap":
            params = re.findall(r"tap\((.*?)\)", act)[0].split(",")
            area = int(params[0].strip())
            subarea = params[1].strip()[1:-1]
            return [act_name + "_grid", area, subarea, last_act]
        elif act_name == "long_press":
            params = re.findall(r"long_press\((.*?)\)", act)[0].split(",")
            area = int(params[0].strip())
            subarea = params[1].strip()[1:-1]
            return [act_name + "_grid", area, subarea, last_act]
        elif act_name == "swipe":
            params = re.findall(r"swipe\((.*?)\)", act)[0].split(",")
            start_area = int(params[0].strip())
            start_subarea = params[1].strip()[1:-1]
            end_area = int(params[2].strip())
            end_subarea = params[3].strip()[1:-1]
            return [act_name + "_grid", start_area, start_subarea, end_area, end_subarea, last_act]
        elif act_name == "grid":
            return [act_name]
        else:
            print_with_color(f"ERROR: Undefined act {act_name}!", "red")
            return ["ERROR"]
    except Exception as e:
        print_with_color(f"ERROR: an exception occurs while parsing the model response: {e}", "red")
        print_with_color(rsp, "red")
        return ["ERROR"]


def parse_reflect_rsp(rsp):
    try:
        decision = re.findall(r"Decision: (.*?)$", rsp, re.MULTILINE)[0]
        think = re.findall(r"Thought: (.*?)$", rsp, re.MULTILINE)[0]
        print_with_color("Decision:", "yellow")
        print_with_color(decision, "magenta")
        print_with_color("Thought:", "yellow")
        print_with_color(think, "magenta")
        if decision == "INEFFECTIVE":
            return [decision, think]
        elif decision == "BACK" or decision == "CONTINUE" or decision == "SUCCESS":
            doc = re.findall(r"Documentation: (.*?)$", rsp, re.MULTILINE)[0]
            print_with_color("Documentation:", "yellow")
            print_with_color(doc, "magenta")
            return [decision, think, doc]
        else:
            print_with_color(f"ERROR: Undefined decision {decision}!", "red")
            return ["ERROR"]
    except Exception as e:
        print_with_color(f"ERROR: an exception occurs while parsing the model response: {e}", "red")
        print_with_color(rsp, "red")
        return ["ERROR"]
