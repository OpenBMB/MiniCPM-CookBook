"""本文件是主文件，修改48行到68行的参数，然后运行即可"""

from prompt_plamte import (
    task_check_template,
    task_split_template,
    task_combine_template,
)
from fuctions import (
    get_check_text,
    get_tools_description,
    get_task_and_question,
    task_text_split,
    bm25,
    distance,
    gpt_35_api_stream,
)
import json
import copy
import math
import os
from argparse import ArgumentParser
from all_param_inference import all_param_split_task
import warnings

warnings.filterwarnings("ignore")
from lora_inference_nomerge import get_merge_model, split_task
import gradio as gr
import json5
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

# from fastbm25 import fastbm25
from sentence_transformers import SentenceTransformer
from bing_search import *
import time
import random
from tools_introduction import call_plugin, tools
from load_model import get_model
from prompt_plamte import (
    TOOL_DESC,
    PROMPT_REACT,
    check_action_inputs,
    prompt_task_split,
    tool_examples,
)
import re


import logging
import time

# 配置日志记录器
from datetime import datetime

# 获取当前时间
now = datetime.now()

# 格式化时间
formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
# logging.basicConfig(
#     filename="/ai/ld/remote/Qwen-main/get_subtask/expriment/example"
#     + formatted_time
#     + ".log",
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
# )
# 训练格式，默认false就好
template_new = False


def _get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--prompt_task_split",
        type=bool,
        default=False,  # 是否分解任务
        help="是否任务分解",
    )
    parser.add_argument(
        "--check_split", default=False, help="是否进行任务检查"
    )  # 是否进行任务检查
    parser.add_argument(
        "--orgin_split_task", default=False, help="是否使用原始子任务合成的prompt"
    )  # 是否使用原始子任务合成的prompt
    parser.add_argument(
        "--embeding_model_path",
        default="/root/ld/ld_model_pretrained/bge-m3",
        help="嵌入模型的路径",
    )  # 词嵌入模型的路径，用于文本匹配
    parser.add_argument(
        "--orgin_split_task_chain",
        default=False,
        help="是否使用原始的子任务的列表，且使用chain方式",
    )  # 是否使用原始人工拆分的子任务的列表
    parser.add_argument(
        "--lora_split_task_chain",
        default=False,
        help="是否使用lora模型进行任务分解且使用chain方式",
    )  # 是否使用lora模型进行任务分解
    parser.add_argument(
        "--allparams_split_task_chain",
        default=False,
        help="使用全量训练的模型进行任务分解",
    )  # 使用全量训练的模型进行任务分解
    parser.add_argument(
        "--execute_model_path",
        default="/root/ld/ld_project/pull_request/MiniCPM-V/finetune/output/merge",
        help="执行模块基座模型",
    )  # 执行模块基座模型
    parser.add_argument(
        "--execute_reflexion", default=True, help="是否使用反思模式"
    )  # 是否使用反思模式
    args = parser.parse_args()
    return args


args = _get_args()
# assert args.prompt_task_split and  args.allparams_split_task_chain and args.lora_split_task_chain == False,'任务分解智能选择一种模式，不能并存'
model, merge_model, embeding_model, tokenizer, merge_tokenizer = get_model(
    args
)  # MODEL是用来执行任务的，merge_model是用来规划任务的


# 主函数，里面包括了各种模式的任务分解和任务执行以及错误反思
def llm_with_plugin(
    prompt: str,
    history,
    args,
    list_of_plugin_info=(),
    write_file=None,
    embeding_model=None,
    orgin_question=None,
):
    # chat_history = [(x['user'], x['bot']) for x in history] + [(prompt, '')]
    # 实验阶段注意修改args.orgin_split_task_chain和task_switch两个参数
    question = prompt  # 原始问题

    # 以下在进行各种模式的任务分解
    if args.orgin_split_task_chain == True:  # 如果使用原始子任务且使用chain的方式
        task_switch = True
        question_subtask_dict = get_task_and_question(
            "/ai/ld/remote/Qwen-main/get_subtask/data_process/all_data.txt"
        )
        subtask = question_subtask_dict[prompt]
        print("任务分解为:", subtask)
        if len(subtask) == 1 and "提供的工具作用较小" in subtask[0]:  # 不使用工具问题
            task_switch = False
            prompt = "Thought:提供的工具作用较小，我将直接回答" + prompt
        elif (
            len(subtask) == 1 and "提供的工具作用较小" not in subtask[0]
        ):  # 单链条使用工具问题
            prompt = subtask[0]
        else:
            subtask.append("输出{}任务得最终结果".format(prompt))
            prompt = "\n".join([str(i + 1) + sub for i, sub in enumerate(subtask)])
    elif (
        args.lora_split_task_chain or args.allparams_split_task_chain
    ):  # 如果用lora或者全量分解任务
        task_switch = True
        merge_model.generation_config.eos_token_id = [2512, 19357, 151643]
        if args.lora_split_task_chain:
            subtask = split_task(prompt, tokenizer, merge_model)  # lora分解任务
        else:
            subtask = all_param_split_task(
                prompt, tokenizer, merge_model
            )  # 全量分解任务
        print("任务分解为:", subtask)
        if len(subtask) == 1 and "提供的工具作用较小" in subtask[0]:  # 不使用工具问题
            task_switch = False
            prompt = "Thought:提供的工具作用较小，我将直接回答" + prompt
        elif (
            len(subtask) == 1 and "提供的工具作用较小" not in subtask[0]
        ):  # 单链条使用工具问题
            prompt = subtask[0]
        else:
            subtask.append("输出{}任务得最终结果".format(prompt))
            prompt = "\n".join([str(i + 1) + sub for i, sub in enumerate(subtask)])
    elif args.prompt_task_split == True:  # 判断是否需要划分任务
        subtask = prompt_task_split(
            prompt, list_of_plugin_info, args=args, write_file=write_file
        )
        if subtask != prompt:
            task_switch = True
            subtask.append("输出{}任务得最终结果".format(prompt))
            prompt = "\n".join([str(i + 1) + sub for i, sub in enumerate(subtask)])
        else:
            task_switch = False
    else:  # 不进行任务分解
        subtask = None
        task_switch = False
    # 需要让模型进行续写的初始文本
    chat_history = [(x["user"], x["bot"]) for x in history] + [(question, "")]
    planning_prompt = build_input_text(chat_history, list_of_plugin_info)

    text = ""
    count = 0
    while True:
        # 以下是模型的输出
        # text_completion是使用根据目前信息续写action和action_input
        if (
            task_switch
        ):  # 如果进行了任务分解，说明已经开始完成子任务，并且最好是让千问输出一次上一个子任务的结果
            if count != 0 and count < len(subtask) - 1:
                text = text + "根据之前任务的结果，现在应该完成{}这个任务.".format(
                    subtask[count]
                )
                output = text_completion(
                    planning_prompt + text,
                    stop_words=["Observation:", "Observation:\n"],
                )
            elif count >= len(subtask) - 1:
                text = text + "{}.".format(subtask[-1])
                output = text_completion(
                    planning_prompt + text,
                    stop_words=["Observation:", "Observation:\n"],
                )
            elif count == 0:  # 第一个任务
                text = (
                    text
                    + "Thought:最终任务是{}，执行顺序是是{}，当前任务是：{}.".format(
                        question,
                        "\n".join(
                            [str(i + 1) + sub for i, sub in enumerate(subtask[:-1])]
                        ),
                        subtask[count],
                    )
                )
                output = text_completion(
                    planning_prompt + text,
                    stop_words=["Observation:", "Observation:\n"],
                )
        else:  # 简单任务的分支
            output = text_completion(
                planning_prompt + text, stop_words=["Observation:", "Observation:\n"]
            )

        action, action_input, output = parse_latest_plugin_call(output)

        if subtask != None and count > 2 * len(subtask) + 3:
            break
        if "Final Answer" in output:  # 生成结束，并且不再需要调用插件
            text += output
            print("#############结束了############")
            break
        if action:  # 需要调用插件
            # action、action_input 分别为需要调用的插件代号、输入参数
            # observation是插件返回的结果，为字符串
            if subtask != None:
                current_subtask = subtask[count]
            else:
                current_subtask = None
            observation = call_plugin(
                action,
                action_input,
                write_file=write_file,
                embeding_model=embeding_model,
                model=model,
                tokenizer=tokenizer,
                incontext=text,
                subtask=current_subtask,
            )

            # 以下对工具选错和执行失败的进行反思处理
            if args.execute_reflexion:
                if observation == "没有找到该工具":
                    output = text_completion(
                        planning_prompt
                        + text
                        + "请注意{}这个工具不存在，不要使用这个工具".format(action),
                        stop_words=["Observation:", "Observation:\n"],
                    )
                    action, action_input, output = parse_latest_plugin_call(output)
                    observation = call_plugin(
                        action,
                        action_input,
                        write_file=write_file,
                        embeding_model=embeding_model,
                        model=model,
                        tokenizer=tokenizer,
                        incontext=text,
                        subtask=current_subtask,
                    )
                elif observation == "执行失败":
                    print("工具{}的参数{}有误,重新思考".format(action, action_input))
                    # Referential=model.chat(tokenizer,text+'\n\n\n按照{武器A:直升机，位置B:[20,30]}的格式将以上文本中所有的代指A,B,C,D等字母指代的具体值输出成一个字典，不要输出无关的字符。',history=[])[0]
                    output = text_completion(
                        planning_prompt
                        + text
                        + "请注意{}这个参数有误，不能运行".format(action_input),
                        stop_words=["Observation:", "Observation:\n"],
                    )
                    action, action_input, output = parse_latest_plugin_call(output)
                    # action_input=check_action_inputs(subtask[count]+'仅输出action_inputs,请将action_inputs中字母取值如下:'+Referential,action,list_of_plugin_info,text,model=model,tokenizer=tokenizer)
                    observation = call_plugin(
                        action,
                        action_input,
                        write_file=write_file,
                        embeding_model=embeding_model,
                        model=model,
                        tokenizer=tokenizer,
                        incontext=text,
                        subtask=current_subtask,
                    )
            output += f"\nObservation: {observation}\nThought:"
            text += output
        count += 1
    new_history = []
    new_history.extend(history)
    new_history.append({"user": question, "bot": text})
    return text, new_history, prompt


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
        if plugin_info.get("args_format", "json") == "json":
            tool += " Format the arguments as a JSON object."
        elif plugin_info["args_format"] == "code":
            tool += " Enclose the code within triple backticks (`) at the beginning and end of the code."
        else:
            raise NotImplementedError
        tools_text.append(tool)
    tools_text = "\n\n".join(tools_text)

    # 候选插件的代号
    tools_name_text = ", ".join(
        [plugin_info["name_for_model"] for plugin_info in list_of_plugin_info]
    )
    if not template_new:
        im_start = "<|im_start|>"
        im_end = "<|im_end|>"
        prompt = f"{im_start}system\nYou are a helpful assistant.{im_end}"
    else:
        im_start = ""
        im_end = ""
        prompt = ""
    for i, (query, response) in enumerate(chat_history):
        if list_of_plugin_info:  # 如果有候选插件
            # 倒数第一轮或倒数第二轮对话填入详细的插件信息，但具体什么位置填可以自行判断
            if (len(chat_history) == 1) or (i == len(chat_history) - 2):
                query = PROMPT_REACT.format(
                    tools_text=tools_text,
                    tools_name_text=tools_name_text,
                    query=query,
                )
        query = query.lstrip(
            "\n"
        ).rstrip()  # 重要！若不 strip 会与训练时数据的构造方式产生差异。
        response = response.lstrip(
            "\n"
        ).rstrip()  # 重要！若不 strip 会与训练时数据的构造方式产生差异。
        # 使用续写模式（text completion）时，需要用如下格式区分用户和AI：
        prompt += f"\n{im_start}user\n{query}{im_end}"
        prompt += f"\n{im_start}assistant\n{response}{im_end}"
    if not template_new:
        assert prompt.endswith(f"\n{im_start}assistant\n{im_end}")
        prompt = prompt[: -len(f"{im_end}")]
    return prompt


def text_completion(input_text: str, stop_words) -> str:  # 作为一个文本续写模型来使用
    if not template_new:
        im_end = "<|im_end|>"
        if im_end not in stop_words:
            stop_words = stop_words + [im_end]
    else:
        im_end = ""
    stop_words_ids = [tokenizer.encode(w) for w in stop_words]

    # TODO: 增加流式输出的样例实现
    input_ids = torch.tensor([tokenizer.encode(input_text)]).to(model.device)
    output = model.llm.generate(input_ids, eos_token_id=tokenizer.encode("<|im_end|>"),max_length=4096)
    output = output.tolist()[0]
    output = tokenizer.decode(output, errors="ignore")
    assert output.startswith(input_text)
    output = output[len(input_text) :].replace("<|endoftext|>", "").replace(im_end, "")

    for stop_str in stop_words:
        idx = output.find(stop_str)
        if idx != -1:
            output = output[: idx + len(stop_str)]
    return output  # 续写 input_text 的结果，不包含 input_text 的内容


def parse_latest_plugin_call(text):
    plugin_name, plugin_args = "", ""
    if text.startswith("\nAction:"):
        i = text.rfind("\nAction:")
        action_string = "\nAction:"
    else:
        i = text.find("Action:")
        action_string = "Action:"
    if "\nAction Input:" in text:
        j = text.rfind("\nAction Input:")
        action_input_string = "\nAction Input:"
    else:
        j = text.rfind("Action Input:")
        action_input_string = "Action Input:"
    if "\nObservation:" in text:
        k = text.rfind("\nObservation:")
        observation_string = "\nObservation:"
    else:
        k = text.rfind("Observation:")
        observation_string = "Observation:"
    if 0 <= i < j:  # If the text has `Action` and `Action input`,
        if k < j:  # but does not contain `Observation`,
            # then it is likely that `Observation` is ommited by the LLM,
            # because the output text may have discarded the stop word.
            text = text.rstrip() + observation_string  # Add it back.
        k = text.rfind(observation_string)
        plugin_name = text[i + len(action_string) : j].strip()
        plugin_args = text[j + len(action_input_string) : k].strip()
        text = text[:k]
    return plugin_name, plugin_args, text


def test():

    history = []
    args = _get_args()

    def gradio_response(message, history):
        try:
            global history_copy
            if message == "clear":
                history = []
            # message是原始问题
            if history != []:
                history = copy.deepcopy(history_copy)
            print(history)
            response, history, subtaks = llm_with_plugin(
                prompt=message,
                history=[],
                args=args,
                list_of_plugin_info=tools,
                embeding_model=embeding_model,
            )
            history_copy = copy.deepcopy(history)
            subtaks = re.sub("\n", "\\n", subtaks)  # 这个subtaks其实是任务分解结果
            response = re.sub("\n", "\\n", response)
            return (
                "任务分解完成，结果为：\n\n"
                + subtaks
                + "\n\n执行结果为:\n\n"
                + response
            )
        except Exception as e:
            return "出现错误{},请重新输入问题".format(str(e))

    # response, history,prompt = llm_with_plugin(prompt=query, history=[], args=args,list_of_plugin_info=tools,write_file=f,embeding_model=embeding_model,orgin_question=orgin_question[index])
    demo = gr.ChatInterface(
        gradio_response,
        chatbot=gr.Chatbot(label="Chagent:\nchain of agent", height=700),
    )
    demo.launch(share=True)


if __name__ == "__main__":
    test()
