import re
from fastbm25 import fastbm25
import math
from rank_bm25 import BM25Okapi  
import openai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from bing_search import *

def gpt_35_api_stream(messages: list):
    input_prompt="请你现在模仿一个搜索引擎，用最简短的话语回复以下关键词：{}"
    # openai.log = "debug"
    openai.api_key = "sk-pd4kVZOzY3xWw4xL0ALayEjjWBJ0kLUGT4gXvAymsGMDr7YM"
    openai.api_base = "https://api.chatanywhere.com.cn/v1"
    """为提供的对话消息创建新的回答 (流式传输)

    Args:
        messages (list): 完整的对话消息
        api_key (str): OpenAI API 密钥

    Returns:
        tuple: (results, error_desc)
    """
    try:
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=messages,
            stream=True,
        )
        completion = {'role': '', 'content': ''}
        for event in response:
            if event['choices'][0]['finish_reason'] == 'stop':
                print(f'收到的完成数据: {completion}')
                break
            for delta_k, delta_v in event['choices'][0]['delta'].items():
                #print(f'流响应数据: {delta_k} = {delta_v}')
                completion[delta_k] += delta_v
        messages.append(completion)  # 直接在传入参数 messages 中追加消息
        return messages[1]['content']
    except Exception as err:
        return '没有相关消息'
def bm25(query, corpus,model=None):  
    # 创建BM25模型  
    if model is None:
        model = SentenceTransformer(r'/ai/ld/pretrain/bge-base-zh')  # 替换为您选择的中文预训练模型
     # 将待匹配项和匹配项转换为向量表示
    doc_embeddings = model.encode(corpus, convert_to_tensor=True).cpu()
    query_embedding = model.encode([query], convert_to_tensor=True).cpu()
     # 计算待匹配项和匹配项之间的余弦相似度
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    if max(similarities)<0.3:
        return '没有相关消息'
     # 找到最接近的文本
    sorted_indices = list(np.argsort(similarities)[::-1])
    #closest_index = similarities.argmax()
    retrival_list = [corpus[i] for i in sorted_indices[0:5]]
    bm = fastbm25(retrival_list)
    try:
        closest_text=bm.top_k_sentence(query)[0][0]
    except:
        closest_text=retrival_list[0]
    return closest_text
#获取送入子任务检测的规范格式子任务
def get_check_text(sub_task_list):
    check_subtask_text='\ninitial_sutask:'.join(sub_task_list)
    check_subtask_text+='\ncheck_result:'
    return check_subtask_text
#获得工具的基本描述:tools_description
def get_tools_description(tools):
    tools_description=''
    for i in tools:
        """{'name_for_human': '谷歌搜索',
        'name_for_model': 'google_search',
        'description_for_model': '谷歌搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。'},"""
        tools_description+='\n'+i['name_for_model']+':'+i['description_for_model']
    tools_description+='\n'
    return tools_description

def get_task_and_question(path):
    with open(path,'r',encoding='utf-8') as f:
        all_lines=f.readlines()
    print('数据读取的前三行是:')
    print(all_lines[0:3])
    questions_and_subtask={}
    for line in all_lines:
        if line.isspace():
            continue
        elif line.startswith('问题') or line.startswith('question'):
            if ':' in line[:10]:
                last_question=line.split(':')[1].strip()
            elif '：'in line[:10]:
                last_question=line.split('：')[1].strip()
            questions_and_subtask[last_question]=[]
        else:
            if ':' in line[:10]:
                questions_and_subtask[last_question].append(line.split(':')[1].strip())
            elif '：'in line[:10]:
                 questions_and_subtask[last_question].append(line.split('：')[1].strip())
    return questions_and_subtask

def task_text_split(text,question_orgin):#将任务分解后的输出变成任务列表，这里是用了qwen第一次thought后的任务分解
    pattern = r"Complex issue: (Yes|No)\n((?:Subtask: .+\n?)*)"
    matches = re.findall(pattern, text)


    last_complex_issue = matches[-1][0]
    if last_complex_issue == "Yes":
        subtasks = re.findall(r"Subtask: (.+)", matches[-1][1])
        return (subtasks)
    else:
        return (question_orgin)
    

def distance(query,map_dict):#计算距离
    distance_dict={}
    query_coordinate=map_dict[query]
    for weapon,item in map_dict.items():
        if weapon==query:
            continue
        else:
            distance_dict[weapon]=str(round(math.sqrt((float(query_coordinate[0])-float(item[0]))**2+(float(query_coordinate[1])-float(item[1]))**2),1))+'km'
    return [query,distance_dict]

if __name__=='__main__':
    bm25('质量',list({'飞行高度':'0.3km以内','携带武器':'火箭弹','克制武器':'对空导弹','重量':'3000kg',"速度":"100km/h","射程":"2km",'适应场景':'空战','续航':'500km','满载人数':'7人','承载重量':'10000kg','续航里程':'1000km'}.keys()))