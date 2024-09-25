from vllm import LLM, SamplingParams
import argparse
import json
import pandas as pd
from transformers import AutoTokenizer
import re,os

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="/root/ld/ld_model_pretrained/qwen2-72b-instruct-gptq-int4")
parser.add_argument("--out_xlsx", type=str, default='/root/ld/ld_project/law_project/second/qwen_gptq.xlsx')
parser.add_argument("--out_jsonl", type=str, default='/root/ld/ld_project/law_project/second/-50_-5gptq.json')
parser.add_argument("--out_num", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=1)
args = parser.parse_args()


def save_to_jsonl(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for entry in data:
            file.write(json.dumps(entry,ensure_ascii=False) + '\n')  # 写入每一条数据，并在每条数据后加上换行符
def save_to_excel(data, output_file):
    df = pd.DataFrame(data)  # 将列表转换为 DataFrame
    df.to_excel(output_file, index=False)  # 保存为 Excel 文件，不包含索引列

tokenizer=AutoTokenizer.from_pretrained(args.model_path)




def process_prompt(prompt):
    format_qwen="""You are a helpful assistant.<|im_end|>
    <|im_start|>user
    {prompt}<|im_end|>"""
    return format_qwen.format(prompt=prompt)
prompt="""Answer the following questions as best you can. You have access to the following tools:\n\ngoogle_search: Call this tool to interact with the 谷歌搜索 API. What is the 谷歌搜索 API useful for? 谷歌搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。 Parameters: [{\"name\": \"search_query\", \"description\": \"搜索关键词或短语\", \"required\": true, \"schema\": {\"type\": \"string\"}}] Format the arguments as a JSON object.\n\nmilitary_information_search: Call this tool to interact with the 军事情报搜索 API. What is the 军事情报搜索 API useful for? 军事情报搜索是一个通用搜索引擎，可用于访问军事情报网、查询军网、了解军事新闻等。 Parameters: [{\"name\": \"search_query\", \"description\": \"搜索关键词或短语\", \"required\": true, \"schema\": {\"type\": \"string\"}}] Format the arguments as a JSON object.\n\naddress_book: Call this tool to interact with the 通讯录 API. What is the 通讯录 API useful for? 通讯录是用来获取个人信息如电话、邮箱地址、公司地址的软件。 Parameters: [{\"name\": \"person_name\", \"description\": \"被查询者的姓名\", \"required\": true, \"schema\": {\"type\": \"string\"}}] Format the arguments as a JSON object.\n\nQQ_Email: Call this tool to interact with the qq邮箱 API. What is the qq邮箱 API useful for? qq邮箱是一个可以用来发送合接受邮件的工具 Parameters: [{\"E-mail address\": \"E-mail address\", \"description\": \"对方邮箱的地址 发给对方的内容\", \"required\": true, \"schema\": {\"type\": \"string\"}}, {\"E-mail content\": \"E-mail_content\", \"description\": \"发给对方的内容\", \"required\": true, \"schema\": {\"type\": \"string\"}}] Format the arguments as a JSON object.\n\nimage_gen: Call this tool to interact with the 文生图 API. What is the 文生图 API useful for? 文生图是一个AI绘画（图像生成）服务，输入文本描述，返回根据文本作画得到的图片的URL Parameters: [{\"name\": \"prompt\", \"description\": \"英文关键词，描述了希望图像具有什么内容\", \"required\": true, \"schema\": {\"type\": \"string\"}}] Format the arguments as a JSON object.\n\nSituation_display: Call this tool to interact with the 态势显示 API. What is the 态势显示 API useful for? :态势显示是通过输入目标位置坐标和显示范围，从而显示当前敌我双方的战场态势图像，并生成图片 Parameters: [{\"coordinate\": \"[coordinate_x,coordinate_y]\", \"description\": \"目标位置的x和y坐标\", \"required\": true, \"schema\": {\"type\": \"string\"}}, {\"radio\": \"radio\", \"description\": \"态势图像显示的范围半径,单位是km,默认值为300km\", \"required\": true, \"schema\": {\"type\": \"string\"}}] Format the arguments as a JSON object.\n\ncalendar: Call this tool to interact with the 万年历 API. What is the 万年历 API useful for? 万年历获取当前时间的工具 Parameters: [{\"time\": \"time_query\", \"description\": \"目标的地点\", \"location\": \"location_query\", \"required\": true, \"schema\": {\"type\": \"string\"}}] Format the arguments as a JSON object.\n\nmap_search: Call this tool to interact with the 地图 API. What is the 地图 API useful for? 地图是一个可以查询地图上所有单位位置信息的工具，返回所有敌军的位置信息。 Parameters: [{\"lauch\": \"yes\", \"description\": \"yes代表启用地图搜索\", \"required\": true, \"schema\": {\"type\": \"string\"}}] Format the arguments as a JSON object.\n\nknowledge_graph: Call this tool to interact with the 知识图谱 API. What is the 知识图谱 API useful for? 知识图谱是输入武器种类获取该武器的属性，也可以输入某种属性获得所有武器的该属性 Parameters: [{\"weapon\": \"weapon_query\", \"description\": \"武器名称,比如飞机、坦克,所有武器\", \"required\": true, \"schema\": {\"type\": \"string\"}}, {\"attribute\": \"attribute\", \"description\": \"输出武器的该属性：射程/速度/重量/适应场景/克制武器/所有属性\", \"required\": true, \"schema\": {\"type\": \"string\"}}] Format the arguments as a JSON object.\n\nweapon_launch: Call this tool to interact with the 武器发射按钮 API. What is the 武器发射按钮 API useful for? 武器发射按钮是可以启动指定武器打击指定目标位置工具。 Parameters: [{\"weapon_and_coordinate\": [\"weapon_query\", \"target_name\", [\"x\", \"y\"]], \"description\": \"被启动的武器名称 被打击的目标名称 被打击目标的坐标地点\", \"required\": true, \"schema\": {\"type\": \"string\"}}] Format the arguments as a JSON object.\n\ndistance_calculation: Call this tool to interact with the 距离计算器 API. What is the 距离计算器 API useful for? 可以根据目标单位和地图api查询的位置信息，计算出地图上所有其他单位与目标单位的距离 Parameters: [{\"target_and_mapdict\": {\"weapon_query\": [\"x1\", \"y1\"], \"unit2\": [\"x2\", \"y2\"], \"unit3\": [\"x3\", \"y3\"], \"unit4\": [\"x4\", \"y4\"]}, \"description\": \"包括目标单位在内的所有地图上单位的名称和位置参数:{被计算的单位名称:[该单位的x坐标,该单位的y坐标],被计算的另外一个单位名称:[该单位的x坐标,该单位的y坐标],地图上的其他单位名称(可省略):[该单位的x坐标,该单位的y坐标](可省略)}\", \"required\": true, \"schema\": {\"type\": \"string\"}}] Format the arguments as a JSON object.\n\nUse the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [google_search, military_information_search, address_book, QQ_Email, image_gen, Situation_display, calendar, map_search, knowledge_graph, weapon_launch, distance_calculation]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can be repeated zero or more times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\nBegin!\n\nQuestion:"""
question = "请问我应该使用哪种武器打击地方坦克？"
llm = LLM(model=args.model_path, tensor_parallel_size=8, dtype='bfloat16',trust_remote_code=True,enforce_eager=True,gpu_memory_utilization=1)

params_dict = { 
    "n": 1,
    "best_of": 1,
    "presence_penalty": 1.5,    
    "frequency_penalty": 1.0,
    "temperature": 0.8,
    "top_p": 0.8,
    "top_k": -1,
    "use_beam_search": False,
    "length_penalty": 1,
    "early_stopping": False,
    "stop": None,
    "stop_token_ids": None,
    "ignore_eos": False,
    "max_tokens": 1000,
    "logprobs": None,
    "prompt_logprobs": None,
    "skip_special_tokens": True,
}

# Create a sampling params object.
sampling_params = SamplingParams(**params_dict)
# Create an LLM.

outputs = llm.generate(prompt+question, sampling_params)
for i in range(len(outputs)):
    output=outputs[i].outputs[0].text
    print(output)
    