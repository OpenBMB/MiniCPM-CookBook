import re
from fuctions import get_tools_description,task_text_split
from tools_introduction import tools


#下面这个是根据工具进行任务分解的prompt
task_split_template="""请根据以下工具和工具描述:{tools_description} 照程序代码的逻辑,将复杂问题拆分成多个子任务,非复杂问题无需拆分。确保每一个子任务都是简单任务，不能继续拆分。有充分的信息以解决问题,严格按照以下格式返回：
Complex issue: Yes or No
Subtask: 当前要解决的问题
Subtask: 当前要解决的问题
为了帮助你解决接下来的任务或者问题,我们提供了一些相关的示例。
问题:帮我找到地图上射程最远的武器
Complex issue: Yes
Subtask: 使用地图工具获得所有武器名称汇总A
Subtask: 使用知识图谱工具获得所有武器的射程B
Subtask: 根据B找到地图上所有武器A中射程最远的武器
问题:帮我画出离我方坦克最远的敌军
Complex issue: Yes
Subtask: 使用地图工具获取所有单位的位置坐标A和我方坦克的位置坐标B
Subtask: 使用距离计算工具计算出A中和我方坦克位置B中距离最近的敌方单位C
Subtask: 使用image_gen工具画出C的图片，返回C的图片地址D
问题:告诉我张三的公司和职务？
Complex issue: No
Subtask:
Subtask:
问题:你叫什么名字？
Complex issue: No
Subtask:
Subtask:
请注意，其中一些信息包含噪音，因此你应该谨慎信任。
现在开始回答这个任务或者问题。请直接回答这个问题,不要包含其他不需要的文字。
问题:{question_orgin}"""

#下面这个是根据工具进行任务检查的prompt
task_check_template="""请根据以下工具和工具描述:{tools_description},每个子任务只能依赖之前子任务的结果和一个工具，判断每个子任务是否需要继续分解，严格按照以下格式返回：
initial_sutask:修改前的子任务
initial_sutask:修改前的子任务
check_result: Yes or No
Subtask: 修改后的子任务
Subtask: 修改后的子任务

为了帮助你解决接下来的任务或者问题,我们提供了一些相关的示例。
initial_sutask:使用地图工具获取所有单位的位置坐标A
Complex issue: No
Subtask: 

initial_sutask:使用地图工具获取指挥所的位置坐标A和所有单位的位置坐标B
initial_sutask:使用距离计算工具计算出A中和指挥所位置B中距离最近的敌方单位C
initial_sutask:使用image_gen工具画出C的图片，返回C的图片地址D
initial_sutask:使用武器发射按钮工具启动D对C进行打击
initial_sutask:使用image_gen工具画出打击后的结果图片，返回图片地址E
initial_sutask:使用QQ邮箱工具将图片E发送给刘丹
check_result: Yes
Subtask: 使用地图工具获取所有单位的位置坐标A和我方坦克的位置坐标B
Subtask: 使用距离计算工具计算出A中和我方坦克位置B中距离最近的敌方单位C
Subtask: 使用image_gen工具画出C的图片，返回C的图片地址D
Subtask: 使用武器发射按钮工具启动D对C进行打击
Subtask: 使用image_gen工具画出打击后的结果图片，返回图片地址E
Subtask: 使用通讯录工具获取秦龙的邮箱地址F
Subtask: 使用邮箱工具发送结果图片E到邮箱地址F

initial_sutask:使用地图工具获取地图上所有单位坐标A和指挥所坐标B
initial_sutask:使用计算距离工具获取所有单位坐标A中和指挥所坐标B之间距离最近的单位C
check_result: No
Subtask:
Subtask:

请注意，其中一些信息包含噪音，因此你应该谨慎信任。
现在开始回答这个任务或者问题。请直接回答这个问题,不要包含其他不需要的文字。
{subtasks_orgin}"""


task_combine_template="""请根据以下工具和工具描述:{tools_description},生成一些问题以及这些问题如何被解决的步骤，请确保问题需要使用一个或者多个工具能够被回答,不要使用不在工具描述范围内的工具,以下是一些实例帮助你完成这些问题：
生成的格式如下：
问题:这里是根据工具描述生成的问题
Subtask: 这里是以上问题需要使用工具完成的子任务1
Subtask: 这里是以上问题需要使用工具完成的子任务2

问题:帮我找到地图上射程最远的武器
Subtask: 使用地图工具获得所有武器名称汇总A
Subtask: 使用知识图谱工具获得A中各武器的射程集合B
Subtask: 在B中计算射程最远的C
问题:帮我画出离我方坦克最远的敌军
Subtask: 使用地图工具获取所有单位的位置坐标A和我方坦克的位置坐标B
Subtask: 使用距离计算工具计算出A中和我方坦克位置B中距离最近的敌方单位C
Subtask: 使用image_gen工具画出C的图片，返回C的图片地址D
问题:请用邮箱帮我发送“你好”两个字给秦龙
Subtask: 使用通讯录工具找到秦龙的邮箱地址A
Subtask: 使用邮箱向秦龙的地址A发送"你好"两个字


请注意，其中一些信息包含噪音，因此你应该谨慎信任。
现在开始回答这个任务或者问题。请直接回答这个问题,不要包含其他不需要的文字。
问题:"""

#以下是反思模块，用于检查输入参数是否错误
def check_action_inputs(question,action,list_of_plugin_info,history=None,model=None,tokenizer=None):
    print('正在对{action}的参数进行反思'.format(action=action))
    check_prompt='''
    请按照以下示例，根据所选用action及其描述一些示例，选择正确的action_inputs参数：
    question_example:使用distance_calculation计算我方发射阵地1坐标A与敌坦克坐标B的距离\n
    {
            'name_for_human': '距离计算器',
            'name_for_model': 'distance_calculation',
            'description_for_model': '可以根据目标单位和地图api查询的位置信息，计算出地图上所有其他单位与目标单位的距离',
            'parameters': [
                {
                    'target_and_mapdict': ( {'weapon_query':['x1','y1'],'unit2':['x2','y2'],'unit3':['x3','y3'],'unit4':['x4','y4']}),
                    'description': '包括目标单位在内的所有地图上单位的名称和位置参数:{被计算的单位名称:[该单位的x坐标,该单位的y坐标],被计算的另外一个单位名称:[该单位的x坐标,该单位的y坐标],地图上的其他单位名称(可省略):[该单位的x坐标,该单位的y坐标](可省略)}',
                    'required': True,
                    'schema': {'type': 'string'},
                }
            ],
        }
    action_inputs:{"target_and_mapdict": {"我方发射阵地1": [50, 70], "敌坦克": [20, 13]}}
    

    question_example:{question_example}
    {tool}
    action_inputs:{example}
    请注意，其中一些信息包含噪音，因此你应该谨慎信任。
    
    现在请根据以上示例对以下问题中的action_inputs进行补充：
    question_example:{question}
    {tool}
    action_inputs:
    '''
    

    for tool in list_of_plugin_info:
        if tool['name_for_model']==action:
            check_text=re.sub('{question_example}',tool_examples[action]['question'],check_prompt)
            check_text=re.sub('{tool}',str(tool),check_text)
            check_text=re.sub('{example}',tool_examples[action]['action_input'],check_text)
            check_text=re.sub('{question}',question,check_text)
            if history!=None:
                check_text=history+'\n\n\n以上是历史背景信息,根据以上信息为背景,'+check_text
            correct_input=model.chat(tokenizer,check_text,history=[])[0]
            break
    return correct_input



def prompt_task_split(question,tools,args,write_file):#任务分解函数
    tools_description = get_tools_description(tools)  # 工具的描述
    task_split_prompt = task_split_template.format(tools_description=tools_description, question_orgin=question)
    split_text=model.chat(tokenizer,task_split_prompt,history=[])[0]
    
    sub_task=task_text_split(split_text,question)
    if sub_task!=question:
        return sub_task
    else:
        return question
    

# 将一个插件的关键信息拼接成一段文本的模版。
TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters}"""

# ReAct prompting 的 instruction 模版，将包含插件的详细信息。
PROMPT_REACT = """Answer the following questions as best you can. You have access to the following tools:

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

tool_examples={
    'map_search':{'question':'查询王五所在的部队位置坐标B和所有单位的位置信息C。','action_input':'{"lauch": "yes"}'},
    'google_search':{'question':'查询当前时间A的的天气情况B','action_input': '{"search_query": "2023年12月20日 17:34:34 天气情况"}'},
    'military_information_search':{'question':'查询敌方最新坦克的信息','action_input': '{"search_query": "敌方最新型坦克详细信息"}'},
    'address_book':{'question':'获取王五的邮箱地址','action_input':'{"person_name": "王五"}'},
    'QQ_Email':{'question':'发送当前明天早上八点开会到403644786@qq.com','action_input':'{"E-mail_address": "403644786@qq.com", "E-mail_content": "明天早上八点开会"}'},
    'image_gen':{'question':'画一幅可爱的小狗在草地上玩耍的图片','action_input':'{"prompt": "一只可爱的小狗在草地上玩耍"}'},
    'Situation_display':{'question':'显示我方指挥所100km范围内的态势图片','action_input': '{"coordinate": "[0,2]", "radio": "175.4"}'},
    'python_math':{'question':'计算我方发射阵地1的300人和直升机的承载人数7计算需要的疏散次数C','action_input':'{"math_formulation": "300//7+300%7>0"}'},
    'calendar':{'question':'获取当前时间','action_input':'{}'},
    'knowledge_graph':{'question':'查询直升机的克制武器','action_input':'{"weapon": "直升机", "attribute": "克制武器"}'},
    'weapon_launch':{'question':'使用火箭炮打击敌指挥中心','action_input':'{"weapon_and_coordinate": ["火箭炮", "敌指挥中心", [200, 100]]}'},
    'distance_calculation':{'question':'','action_input':'{"target_and_mapdict": {"我方发射阵地1": [50, 70], "敌坦克": [20, 13]}}'}
}
if __name__=='__main__':
    from transformers import AutoTokenizer,AutoModelForCausalLM
    model=AutoModelForCausalLM.from_pretrained('/ai/ld/pretrain/Qwen-72B-Chat-Int4/',  device_map="cuda:0",trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained('/ai/ld/pretrain/Qwen-72B-Chat-Int4/',trust_remote_code=True)
    history='''Thought:最终任务是请告诉我，要对敌直升机进行打击，应选择何种武器，以及需要多少时间飞达？，执行顺序是是1使用knowledge_graph查询敌直升机的克制武器A。
2使用map_search查询敌直升机的位置坐标B。
3使用map_search查询直升机的克制武器A的位置坐标C。
4使用distance_calculation计算克制武器A到敌直升机C的距离D。
5使用knowledge_graph查询克制武器的速度E。
6根据距离D和速度E计算飞行时间F。，当前任务是：使用knowledge_graph查询敌直升机的克制武器A。. 
Action: knowledge_graph
Action Input: {"weapon": "敌直升机", "attribute": "克制武器"}
Observation: 直升机的克制武器是:对空导弹
Thought:根据之前任务的结果，现在应该完成使用map_search查询敌直升机的位置坐标B。这个任务. 
Action: map_search
Action Input: {"lauch": "yes"}
Observation: {'敌直升机': [170, 45], '我方指挥所': [0, 2], '敌坦克': [20, 13], '我方火箭炮': [100, 120], '我方发射阵地1': [50, 70], '我方发射阵地2': [150, 170], '敌指挥中心': [70, 35], '敌反坦克导弹': [50, 100], '我方坦克': [32, 21]}
Thought:根据之前任务的结果，现在应该完成使用map_search查询直升机的克制武器A的位置坐标C。这个任务. 
Action: map_search
Action Input: {"lauch": "yes"}
Observation: {'敌直升机': [170, 45], '我方指挥所': [0, 2], '敌坦克': [20, 13], '我方火箭炮': [100, 120], '我方发射阵地1': [50, 70], '我方发射阵地2': [150, 170], '敌指挥中心': [70, 35], '敌反坦克导弹': [50, 100], '我方坦克': [32, 21]}
Thought:根据之前任务的结果，现在应该完成使用distance_calculation计算克制武器A到敌直升机C的距离D。这个任务. 
Action: distance_calculation
Action Input: {"target_and_mapdict": {"我方发射阵地1": [50, 70], "敌直升机": [170, 45]}}
Observation: 以下是所有单位与我方发射阵地1的距离:{'敌直升机': '122.6km'}
Thought:根据之前任务的结果，现在应该完成使用knowledge_graph查询克制武器的速度E。这个任务. 
Action: knowledge_graph
Action Input: {"weapon": "对空导弹", "attribute": "速度"}
Observation: 火箭炮的速度是:4500km/h
'''
    Referential=model.chat(tokenizer,history+'\n\n\n按照{武器A:直升机，位置B:[20,30]}的格式将以上文本中所有的代指A,B,C,D等字母指代的具体值输出成一个字典格式',history=[])[0]
    check_action_inputs('根据之前任务的结果，现在应该完成根据距离D和速度E计算飞行时间F。','python_math',tools,history,model,tokenizer)
    