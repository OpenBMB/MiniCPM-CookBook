#
# 相关材料：
#   ReAct Prompting 原理简要介绍，不包含代码实现：
#       https://github.com/QwenLM/Qwen-7B/blob/main/examples/react_prompt.md
#   基于 model.chat 接口（对话模式）的 ReAct Prompting 实现（含接入 LangChain 的工具实现）：
#       https://github.com/QwenLM/Qwen-7B/blob/main/examples/langchain_tooluse.ipynb
#   基于 model.generate 接口（续写模式）的 ReAct Prompting 实现，比 chat 模式的实现更复杂些：
#       https://github.com/QwenLM/Qwen-7B/blob/main/examples/react_demo.py（本文件）
#

import json
import os
from PIL import Image
import json5
import torch
from transformers import AutoModel, AutoTokenizer
from transformers.generation import GenerationConfig
from transformers import StoppingCriteria, StoppingCriteriaList, AutoModelForCausalLM, AutoTokenizer
import torch
import warnings
import requests
from io import BytesIO

# 忽略所有警告
warnings.filterwarnings("ignore")
model_path='/root/ld/ld_model_pretrained/Minicpmv2_6'
img_save_path ='/root/ld/ld_project/pull_request/MiniCPM_Series_Tutorial/agent_demo/local_image.jpg'
# 定义自定义的 StoppingCriteria 类
class SequenceStoppingCriteria(StoppingCriteria):
    def __init__(self, sequence_ids):
        self.sequence_ids = sequence_ids
        self.current_sequence = []
    def check_sequences(self, current_tokens, sequences):
        """
        检查当前生成的 tokens 是否包含了特定的连续数字序列。
        
        :param current_tokens: 当前生成的 tokens 列表
        :param sequences: 包含多个连续数字序列的列表
        :return: 如果 current_tokens 中出现了任何序列，则返回 True；否则返回 False
        """
        for i in range(len(current_tokens) - max(map(len, sequences)) + 1):
            for seq in sequences:
                if current_tokens[i:i+len(seq)] == seq:
                    return True
        return False
    def __call__(self, input_ids, scores, **kwargs):
        # 获取当前生成的 tokens
        current_tokens = [input_ids[-1][-1]]

        # 检查连续出现的 tokens 是否匹配停止序列
        self.current_sequence.extend(current_tokens)

        # 检查当前生成的 tokens 是否包含了特定的连续数字序列
        if self.check_sequences(self.current_sequence, self.sequence_ids):
            return True  # 停止生成
        
        return False


for _ in range(10):  # 网络不稳定，多试几次
        name = model_path
        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        generation_config = GenerationConfig.from_pretrained(name, trust_remote_code=True)
        model = AutoModel.from_pretrained(name, trust_remote_code=True,
                attn_implementation='sdpa', torch_dtype=torch.bfloat16)
        model.eval().cuda()

        model.generation_config = generation_config
        model.generation_config.max_length = 4096
        model.generation_config.top_k = 1
        break



# 将一个插件的关键信息拼接成一段文本的模版。
TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters}"""

# ReAct prompting 的 instruction 模版，将包含插件的详细信息。
PROMPT_REACT = """Answer the following questions as best you can. You have access to the following APIs:

{tools_text}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tools_name_text}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {query}"""


#
# 本示例代码的入口函数。
#
# 输入：
#   prompt: 用户的最新一个问题。
#   history: 用户与模型的对话历史，是一个 list，
#       list 中的每个元素为 {"user": "用户输入", "bot": "模型输出"} 的一轮对话。
#       最新的一轮对话放 list 末尾。不包含最新一个问题。
#   list_of_plugin_info: 候选插件列表，是一个 list，list 中的每个元素为一个插件的关键信息。
#       比如 list_of_plugin_info = [plugin_info_0, plugin_info_1, plugin_info_2]，
#       其中 plugin_info_0, plugin_info_1, plugin_info_2 这几个样例见本文档前文。
#
# 输出：
#   模型对用户最新一个问题的回答。
#
def llm_with_plugin(prompt: str, history, list_of_plugin_info=()):
    chat_history = [(x['user'], x['bot']) for x in history] + [(prompt, '')]

    # 需要让模型进行续写的初始文本
    planning_prompt = build_input_text(chat_history, list_of_plugin_info)

    text = ''
    while True:
        output = text_completion(planning_prompt + text, stop_words=['Observation:', 'Observation:\n'])
        action, action_input, output = parse_latest_plugin_call(output)
        if action:  # 需要调用插件
            # action、action_input 分别为需要调用的插件代号、输入参数
            # observation是插件返回的结果，为字符串
            observation = call_plugin(action, action_input)
            output += f'\nObservation: {observation}\nThought:'
            text += output
        else:  # 生成结束，并且不再需要调用插件
            text += output
            break

    new_history = []
    new_history.extend(history)
    new_history.append({'user': prompt, 'bot': text})
    return text, new_history


# 将对话历史、插件信息聚合成一段初始文本
def build_input_text(chat_history, list_of_plugin_info) -> str:
    # 候选插件的详细信息
    tools_text = []
    for plugin_info in list_of_plugin_info:
        tool = TOOL_DESC.format(
            name_for_model=plugin_info["name_for_model"],
            name_for_human=plugin_info["name_for_human"],
            description_for_model=plugin_info["description_for_model"],
            parameters=json.dumps(plugin_info["parameters"], ensure_ascii=False),
        )
        if plugin_info.get('args_format', 'json') == 'json':
            tool += " Format the arguments as a JSON object."
        elif plugin_info['args_format'] == 'code':
            tool += ' Enclose the code within triple backticks (`) at the beginning and end of the code.'
        else:
            raise NotImplementedError
        tools_text.append(tool)
    tools_text = '\n\n'.join(tools_text)

    # 候选插件的代号
    tools_name_text = ', '.join([plugin_info["name_for_model"] for plugin_info in list_of_plugin_info])

    im_start = '<|im_start|>'
    im_end = '<|im_end|>'
    prompt = f'{im_start}system\nYou are a helpful assistant.{im_end}'
    for i, (query, response) in enumerate(chat_history):
        if list_of_plugin_info:  # 如果有候选插件
            # 倒数第一轮或倒数第二轮对话填入详细的插件信息，但具体什么位置填可以自行判断
            if (len(chat_history) == 1) or (i == len(chat_history) - 2):
                query = PROMPT_REACT.format(
                    tools_text=tools_text,
                    tools_name_text=tools_name_text,
                    query=query,
                )
        query = query.lstrip('\n').rstrip()  # 重要！若不 strip 会与训练时数据的构造方式产生差异。
        response = response.lstrip('\n').rstrip()  # 重要！若不 strip 会与训练时数据的构造方式产生差异。
        # 使用续写模式（text completion）时，需要用如下格式区分用户和AI：
        prompt += f"\n{im_start}user\n{query}{im_end}"
        prompt += f"\n{im_start}assistant\n{response}{im_end}"

    assert prompt.endswith(f"\n{im_start}assistant\n{im_end}")
    prompt = prompt[: -len(f'{im_end}')]
    return prompt


def text_completion(input_text: str, stop_words) -> str:  # 作为一个文本续写模型来使用
    im_end = '<|im_end|>'
    if im_end not in stop_words:
        stop_words = stop_words + [im_end]
    stop_words_ids = [tokenizer.encode(w) for w in stop_words]
    #stop_words_ids = [tokenizer.encode(word, add_special_tokens=False) for word in stop_words]
    #stop_words_ids = [token_id for sublist in stop_words_ids for token_id in sublist]
    stopping_criteria = StoppingCriteriaList([SequenceStoppingCriteria(stop_words_ids)])

    # TODO: 增加流式输出的样例实现
    input_ids = torch.tensor([tokenizer.encode(input_text)]).to(model.device)
    output = model.llm.generate(input_ids, stopping_criteria=stopping_criteria,max_length=4096,do_sample=False)
    output = output.tolist()[0]
    output = tokenizer.decode(output, errors="ignore")
    assert output.startswith(input_text)
    output = output[len(input_text) :].replace('<|endoftext|>', '').replace(im_end, '')

    for stop_str in stop_words:
        idx = output.find(stop_str)
        if idx != -1:
            output = output[: idx + len(stop_str)]
    return output  # 续写 input_text 的结果，不包含 input_text 的内容


def parse_latest_plugin_call(text):
    plugin_name, plugin_args = '', ''
    i = text.rfind('\nAction:')
    j = text.rfind('\nAction Input:')
    k = text.rfind('\nObservation:')
    if 0 <= i < j:  # If the text has `Action` and `Action input`,
        if k < j:  # but does not contain `Observation`,
            # then it is likely that `Observation` is ommited by the LLM,
            # because the output text may have discarded the stop word.
            text = text.rstrip() + '\nObservation:'  # Add it back.
        k = text.rfind('\nObservation:')
        plugin_name = text[i + len('\nAction:') : j].strip()
        plugin_args = text[j + len('\nAction Input:') : k].strip()
        text = text[:k]
    return plugin_name, plugin_args, text


#
# 输入：
#   plugin_name: 需要调用的插件代号，对应 name_for_model。
#   plugin_args：插件的输入参数，是一个 dict，dict 的 key、value 分别为参数名、参数值。
# 输出：
#   插件的返回结果，需要是字符串。
#   即使原本是 JSON 输出，也请 json.dumps(..., ensure_ascii=False) 成字符串。
#
def call_plugin(plugin_name: str, plugin_args: str) -> str:
    #
    # 请开发者自行完善这部分内容。这里的参考实现仅是 demo 用途，非生产用途。
    #
    if plugin_name == 'image_gen_prompt':
        # 使用 SerpAPI 需要在这里填入您的 SERPAPI_API_KEY！
        try:
            image_path = json5.loads(plugin_args)["image_path"]
            if image_path.startswith('http'):
                headers = {
            'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
                yzmdata = requests.get(image_path,headers=headers)
                tempIm = BytesIO(yzmdata.content)
                image1 = Image.open(tempIm).convert('RGB')
                image1.save(img_save_path)
                image1 = Image.open(img_save_path).convert('RGB')
            else:
                image1 = Image.open(image_path).convert('RGB')
        except:
            image_path=input("请输入图片地址或网址：")
            if image_path.startswith('http'):
                headers = {
            'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
                yzmdata = requests.get(image_path,headers=headers)
                tempIm = BytesIO(yzmdata.content)
                image1 = Image.open(tempIm).convert('RGB')
                image1.save(img_save_path)
                image1 = Image.open(img_save_path).convert('RGB')
            else:
                image1 = Image.open(image_path).convert('RGB')
        question1 = 'Please describe all the details in this picture in detail?'
        msgs = [
            {'role': 'user', 'content': question1},
        ]

        res = model.chat(
            image=image1,
            msgs=msgs,
            tokenizer=tokenizer
        )
        return res
    elif plugin_name == 'image_gen':
        import urllib.parse
        prompt = json5.loads(plugin_args)["prompt"]
        prompt = urllib.parse.quote(prompt)
        return json.dumps({'image_url': f'https://image.pollinations.ai/prompt/{prompt}'}, ensure_ascii=False)
    elif plugin_name == 'Modify_text':
        import urllib.parse
        prompt_input = json5.loads(plugin_args)["describe_before"]
        Modification_request = json5.loads(plugin_args)["Modification_request"]
        input_prompt = "请将以下的prompt:{}按照以下要求修改:{}.修改后的prompt:".format(prompt_input,Modification_request)
        im_start = '<|im_start|>'
        im_end = '<|im_end|>'
        prompt = f'{im_start}system\nYou are a helpful assistant.{im_end}'+f"\n{im_start}user\n{input_prompt}{im_end}"
        input_ids = torch.tensor([tokenizer.encode(prompt)]).to(model.device)
        output = model.llm.generate(input_ids, max_length=4096)
        output = output.tolist()[0]
        output = tokenizer.decode(output, errors="ignore")
        return output
    else:
        raise NotImplementedError


def test():
    tools = [
        {
            'name_for_human': '图生文',
            'name_for_model': 'image_gen_prompt',
            'description_for_model': '图生文是一个可以看图生成文字描述的服务，输入一张图片的地址，将返回图片详细逼真的表述',
            'parameters': [
                {
                    'name': 'image_path',
                    'description': '需要图片描述的地址',
                    'required': True,
                    'schema': {'type': 'string'},
                }
            ],
        },
        {
            'name_for_human': '文生图',
            'name_for_model': 'image_gen',
            'description_for_model': '文生图是一个AI绘画（图像生成）服务，输入文本描述，返回根据文本作画得到的图片的URL',
            'parameters': [
                {
                    'name': 'prompt',
                    'description': '英文关键词，描述了希望图像具有什么内容的文字prompt',
                    'required': True,
                    'schema': {'type': 'string'},
                }
            ],
        },
        {
            'name_for_human': '提示语修改',
            'name_for_model': 'Modify_text',
            'description_for_model': '提示语修改是一个根据输入要求将原始prompt修改成为更符合要求的prompt的东西',
            'parameters': [
                {
                    'name': 'describe_before',
                    'description': '未修改之前的提示语或图片描述',
                    'required': True,
                    'schema': {'type': 'string'},
                },
                {
                    'name': 'Modification_request',
                    'description': '对提示语和图片描述的修改要求，比如将文字中的猫改为狗',
                    'required': True,
                    'schema': {'type': 'string'},
                }
            ],
        },
    ]
    history = []
    while True:
    #for query in ['你好', '搜索一下谁是周杰伦', '再搜下他老婆是谁', '给我画个可爱的小猫吧，最好是黑猫']:
        query=input("User's Query:")
        response, history = llm_with_plugin(prompt=query, history=history, list_of_plugin_info=tools)
        print(f"Qwen's Response:\n{response}\n")


if __name__ == "__main__":
    test()