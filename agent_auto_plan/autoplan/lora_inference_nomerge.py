from peft import PeftModel, PeftConfig
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig
import time
import re
import warnings
import copy
warnings.filterwarnings("ignore")
def get_merge_model(base_path,adapter_path):
    max_memory = f"{int(torch.cuda.mem_get_info()[0] / 1024 ** 3) - 2}GB"
    generation_config = GenerationConfig.from_pretrained(base_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
            base_path,
            device_map="cuda:0",
            max_memory=max_memory,
            trust_remote_code=True,
            #use_safetensors=True,
            bf16=True
        ).eval()
    base_model=copy.deepcopy(model)
    model.generation_config = generation_config
    model.generation_config.do_sample = False
    merge_model = PeftModel.from_pretrained(model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    merge_model.generation_config.eos_token_id=[2512,19357,151643]
    return merge_model,base_model,tokenizer
def split_task(question,tokenizer,merge_model,prompt=None):
    if prompt==None:
        prompt= """\nuser\nAnswer the following questions as best you can. You have access to the following tools:\n\ngoogle_search: Call this tool to interact with the 谷歌搜索 API. What is the 谷歌搜索 API useful for? 谷歌搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。 Parameters: [{"name": "search_query", "description": "搜索关键词或短语", "required": true, "schema": {"type": "string"}}] Format the arguments as a JSON object.\n\nmilitary_information_search: Call this tool to interact with the 军事情报搜索 API. What is the 军事情报搜索 API useful for? 军事情报搜索是一个通用搜索引擎，可用于访问军事情报网、查询军网、了解军事新闻等。 Parameters: [{"name": "search_query", "description": "搜索关键词或短语", "required": true, "schema": {"type": "string"}}] Format the arguments as a JSON object.\n\naddress_book: Call this tool to interact with the 通讯录 API. What is the 通讯录 API useful for? 通讯录是用来获取个人信息如电话、邮箱地址、公司地址的软件。 Parameters: [{"name": "person_name", "description": "被查询者的姓名", "required": true, "schema": {"type": "string"}}] Format the arguments as a JSON object.\n\nQQ_Email: Call this tool to interact with the qq邮箱 API. What is the qq邮箱 API useful for? qq邮箱是一个可以用来发送合接受邮件的工具 Parameters: [{"E-mail address": "E-mail address", "description": "对方邮箱的地址 发给对方的内容", "required": true, "schema": {"type": "string"}}, {"E-mail content": "E-mail_content", "description": "发给对方的内容", "required": true, "schema": {"type": "string"}}] Format the arguments as a JSON object.\n\nimage_gen: Call this tool to interact with the 文生图 API. What is the 文生图 API useful for? 文生图是一个AI绘画（图像生成）服务，输入文本描述，返回根据文本作画得到的图片的URL Parameters: [{"name": "prompt", "description": "英文关键词，描述了希望图像具有什么内容", "required": true, "schema": {"type": "string"}}] Format the arguments as a JSON object.\n\nSituation_display: Call this tool to interact with the 态势显示 API. What is the 态势显示 API useful for? :态势显示是通过输入目标位置坐标和显示范围，从而显示当前敌我双方的战场态势图像，并生成图片 Parameters: [{"coordinate": "[coordinate_x,coordinate_y]", "description": "目标位置的x和y坐标", "required": true, "schema": {"type": "string"}}, {"radio": "radio", "description": "态势图像显示的范围半径,单位是km,默认值为300km", "required": true, "schema": {"type": "string"}}] Format the arguments as a JSON object.\n\ncalendar: Call this tool to interact with the 万年历 API. What is the 万年历 API useful for? 万年历获取当前时间的工具 Parameters: [{"time": "time_query", "description": "目标的地点", "location": "location_query", "required": true, "schema": {"type": "string"}}] Format the arguments as a JSON object.\n\nmap_search: Call this tool to interact with the 地图 API. What is the 地图 API useful for? 地图是一个可以查询地图上所有单位位置信息的工具，返回所有敌军的位置信息。 Parameters: [{"lauch": "yes", "description": "yes代表启用地图搜索", "required": true, "schema": {"type": "string"}}] Format the arguments as a JSON object.\n\nknowledge_graph: Call this tool to interact with the 知识图谱 API. What is the 知识图谱 API useful for? 知识图谱是输入武器种类获取该武器的属性，也可以输入某种属性获得所有武器的该属性 Parameters: [{"weapon": "weapon_query", "description": "武器名称,比如飞机、坦克,所有武器", "required": true, "schema": {"type": "string"}}, {"attribute": "attribute", "description": "输出武器的该属性：射程/速度/重量/适应场景/克制武器/所有属性", "required": true, "schema": {"type": "string"}}] Format the arguments as a JSON object.\n\nweapon_launch: Call this tool to interact with the 武器发射按钮 API. What is the 武器发射按钮 API useful for? 武器发射按钮是可以启动指定武器打击指定目标位置工具。 Parameters: [{"weapon_and_coordinate": ["weapon_query", "target_name", ["x", "y"]], "description": "被启动的武器名称 被打击的目标名称 被打击目标的坐标地点", "required": true, "schema": {"type": "string"}}] Format the arguments as a JSON object.\n\ndistance_calculation: Call this tool to interact with the 距离计算器 API. What is the 距离计算器 API useful for? 可以根据目标单位和地图api查询的位置信息，计算出地图上所有其他单位与目标单位的距离 Parameters: [{"target_and_mapdict": {"weapon_query": ["x1", "y1"], "unit2": ["x2", "y2"], "unit3": ["x3", "y3"], "unit4": ["x4", "y4"]}, "description": "包括目标单位在内的所有地图上单位的名称和位置参数:{被计算的单位名称:[该单位的x坐标,该单位的y坐标],被计算的另外一个单位名称:[该单位的x坐标,该单位的y坐标],地图上的其他单位名称(可省略):[该单位的x坐标,该单位的y坐标](可省略)}", "required": true, "schema": {"type": "string"}}] Format the arguments as a JSON object.\n\nUse the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [google_search, military_information_search, address_book, QQ_Email, image_gen, Situation_display, calendar, map_search, knowledge_graph, weapon_launch, distance_calculation]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can be repeated zero or more times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\nBegin!\n\nQuestion: {question}\nassistant\nThought:"""
    prompt_question=re.sub('{question}',question,prompt)
    inputs = tokenizer(prompt_question, return_tensors="pt").to("cuda:0")
    start_time=time.time()
    outputs = merge_model.generate(
            input_ids=inputs["input_ids"], 
            attention_mask=inputs["attention_mask"], 
            #stopping_criteria=stopping_criteria,
            # stop_words_ids=stop_words_ids
        )
    outputs=tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
    end_time=time.time()
    inference_time=end_time-start_time
    outputs=outputs[len(prompt_question):]
    outputs=re.sub('Action+|Final+','',outputs)
    outputs=re.sub('\n+','\n',outputs)
    if outputs[0]==':' or outputs[0]=='：':
        outputs=outputs[1:]
    outputs=outputs.strip()
    outputs=outputs.split('\n')
    outputs = [item.split(".")[1] for item in outputs if item != ""]
    return outputs
def format_outputs(outputs):#将输出格式化
    outputs=outputs[len(prompt_question):]
    outputs=re.sub('Action+|Final+','',outputs)
    outputs=re.sub('\n+','\n',outputs)
    if outputs[0]==':' or outputs[0]=='：':
        outputs=outputs[1:]
    outputs=outputs.strip()
    outputs=outputs.split('\n')
    outputs = [item.split(".")[1] for item in outputs if item != ""]
    return outputs
def test():
    merge_model,model,tokenizer=get_merge_model('/ai/ld/pretrain/Qwen-14B-Chat/','/ai/ld/remote/Qwen-main/output_qwen/checkpoint-200')
    output=split_task('安排步兵单位前往监视敌方发射阵地1，报告监视结果给我方指挥所。',tokenizer,merge_model)
    return output
if __name__ == "__main__":
    #print(test())
    prompt= """\nuser\nAnswer the following questions as best you can. You have access to the following tools:\n\ngoogle_search: Call this tool to interact with the 谷歌搜索 API. What is the 谷歌搜索 API useful for? 谷歌搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。 Parameters: [{"name": "search_query", "description": "搜索关键词或短语", "required": true, "schema": {"type": "string"}}] Format the arguments as a JSON object.\n\nmilitary_information_search: Call this tool to interact with the 军事情报搜索 API. What is the 军事情报搜索 API useful for? 军事情报搜索是一个通用搜索引擎，可用于访问军事情报网、查询军网、了解军事新闻等。 Parameters: [{"name": "search_query", "description": "搜索关键词或短语", "required": true, "schema": {"type": "string"}}] Format the arguments as a JSON object.\n\naddress_book: Call this tool to interact with the 通讯录 API. What is the 通讯录 API useful for? 通讯录是用来获取个人信息如电话、邮箱地址、公司地址的软件。 Parameters: [{"name": "person_name", "description": "被查询者的姓名", "required": true, "schema": {"type": "string"}}] Format the arguments as a JSON object.\n\nQQ_Email: Call this tool to interact with the qq邮箱 API. What is the qq邮箱 API useful for? qq邮箱是一个可以用来发送合接受邮件的工具 Parameters: [{"E-mail address": "E-mail address", "description": "对方邮箱的地址 发给对方的内容", "required": true, "schema": {"type": "string"}}, {"E-mail content": "E-mail_content", "description": "发给对方的内容", "required": true, "schema": {"type": "string"}}] Format the arguments as a JSON object.\n\nimage_gen: Call this tool to interact with the 文生图 API. What is the 文生图 API useful for? 文生图是一个AI绘画（图像生成）服务，输入文本描述，返回根据文本作画得到的图片的URL Parameters: [{"name": "prompt", "description": "英文关键词，描述了希望图像具有什么内容", "required": true, "schema": {"type": "string"}}] Format the arguments as a JSON object.\n\nSituation_display: Call this tool to interact with the 态势显示 API. What is the 态势显示 API useful for? :态势显示是通过输入目标位置坐标和显示范围，从而显示当前敌我双方的战场态势图像，并生成图片 Parameters: [{"coordinate": "[coordinate_x,coordinate_y]", "description": "目标位置的x和y坐标", "required": true, "schema": {"type": "string"}}, {"radio": "radio", "description": "态势图像显示的范围半径,单位是km,默认值为300km", "required": true, "schema": {"type": "string"}}] Format the arguments as a JSON object.\n\ncalendar: Call this tool to interact with the 万年历 API. What is the 万年历 API useful for? 万年历获取当前时间的工具 Parameters: [{"time": "time_query", "description": "目标的地点", "location": "location_query", "required": true, "schema": {"type": "string"}}] Format the arguments as a JSON object.\n\nmap_search: Call this tool to interact with the 地图 API. What is the 地图 API useful for? 地图是一个可以查询地图上所有单位位置信息的工具，返回所有敌军的位置信息。 Parameters: [{"lauch": "yes", "description": "yes代表启用地图搜索", "required": true, "schema": {"type": "string"}}] Format the arguments as a JSON object.\n\nknowledge_graph: Call this tool to interact with the 知识图谱 API. What is the 知识图谱 API useful for? 知识图谱是输入武器种类获取该武器的属性，也可以输入某种属性获得所有武器的该属性 Parameters: [{"weapon": "weapon_query", "description": "武器名称,比如飞机、坦克,所有武器", "required": true, "schema": {"type": "string"}}, {"attribute": "attribute", "description": "输出武器的该属性：射程/速度/重量/适应场景/克制武器/所有属性", "required": true, "schema": {"type": "string"}}] Format the arguments as a JSON object.\n\nweapon_launch: Call this tool to interact with the 武器发射按钮 API. What is the 武器发射按钮 API useful for? 武器发射按钮是可以启动指定武器打击指定目标位置工具。 Parameters: [{"weapon_and_coordinate": ["weapon_query", "target_name", ["x", "y"]], "description": "被启动的武器名称 被打击的目标名称 被打击目标的坐标地点", "required": true, "schema": {"type": "string"}}] Format the arguments as a JSON object.\n\ndistance_calculation: Call this tool to interact with the 距离计算器 API. What is the 距离计算器 API useful for? 可以根据目标单位和地图api查询的位置信息，计算出地图上所有其他单位与目标单位的距离 Parameters: [{"target_and_mapdict": {"weapon_query": ["x1", "y1"], "unit2": ["x2", "y2"], "unit3": ["x3", "y3"], "unit4": ["x4", "y4"]}, "description": "包括目标单位在内的所有地图上单位的名称和位置参数:{被计算的单位名称:[该单位的x坐标,该单位的y坐标],被计算的另外一个单位名称:[该单位的x坐标,该单位的y坐标],地图上的其他单位名称(可省略):[该单位的x坐标,该单位的y坐标](可省略)}", "required": true, "schema": {"type": "string"}}] Format the arguments as a JSON object.\n\nUse the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [google_search, military_information_search, address_book, QQ_Email, image_gen, Situation_display, calendar, map_search, knowledge_graph, weapon_launch, distance_calculation]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can be repeated zero or more times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\nBegin!\n\nQuestion: {question}\nassistant\nThought:"""
    question='判断我方直升机在敌方防空火力覆盖区内飞行的安全性，有哪些方法可用？'
    tokenizer = AutoTokenizer.from_pretrained("/ai/ld/pretrain/Qwen-14B-Chat", trust_remote_code=True)
    #peft_model_id = f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"
    #config = PeftConfig.from_pretrained(peft_model_id)
    # tokenizer编码

    # 加载基础模型
    max_memory = f"{int(torch.cuda.mem_get_info()[0] / 1024 ** 3) - 2}GB"
    generation_config = GenerationConfig.from_pretrained("/ai/ld/pretrain/Qwen-14B-Chat", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
                "/ai/ld/pretrain/Qwen-14B-Chat",
                device_map="cuda:0",
                max_memory=max_memory,
                trust_remote_code=True,
                #use_safetensors=True,
                bf16=True
            ).eval()
    model.generation_config = generation_config
    model.generation_config.do_sample = False
    # stop_words=['\n\n\n', 'Action','Final']
    # stop_words_ids=[tokenizer.encode(w) for w in stop_words]+[[151643]]
    model.generation_config.eos_token_id=[2512,19357,151643]
    #model.generation_config.top_k = 1
    # 加载PEFT模型
    merge_model = PeftModel.from_pretrained(model, '/ai/ld/remote/Qwen-main/output_qwen/checkpoint-400')

    def stopping_criteria(cur_len,output_so_far):
        
        # 停止条件：生成的文本中包含特定的词语
        if "\n\n\n" in output_so_far:
            return True
        return False
    # 模型推理
    while True:
        question=input("请输入问题：")
        if question=="exit":
            break
        prompt_question=re.sub('{question}',question,prompt)
        inputs = tokenizer(prompt_question, return_tensors="pt").to("cuda:0")
        start_time=time.time()
        outputs = merge_model.generate(
                input_ids=inputs["input_ids"], 
                attention_mask=inputs["attention_mask"], 
                #stopping_criteria=stopping_criteria,
                # stop_words_ids=stop_words_ids
            )
        end_time=time.time()
        print(f"推理耗时：{end_time-start_time}s")
        # tokenizer解码
        outputs=tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
        outputs=outputs[len(prompt_question):]
        #print(outputs)
        outputs=re.sub('Action+|Final+','',outputs)
        # pattern = r'首先,|然后,|接下来,|最后,'
        # outputs=re.sub(pattern, '\n', outputs)
        outputs=re.sub('\n+','\n',outputs)
        print(outputs)