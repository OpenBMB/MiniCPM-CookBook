import json5
import bing_search as query_bing
from fuctions import bm25,distance,task_text_split
import re

#下面是工具的介绍和接口定义，可以自行修改
tools = [
        {
            'name_for_human': '谷歌搜索',
            'name_for_model': 'google_search',
            'description_for_model': '谷歌搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。',
            'parameters': [
                {
                    'name': 'search_query',
                    'description': '搜索关键词或短语',
                    'required': True,
                    'schema': {'type': 'string'},
                }
            ]
        },
        {
            'name_for_human': '军事情报搜索',
            'name_for_model': 'military_information_search',
            'description_for_model': '军事情报搜索是一个通用搜索引擎，可用于访问军事情报网、查询军网、了解军事新闻等。',
            'parameters': [
                {
                    'name': 'search_query',
                    'description': '搜索关键词或短语',
                    'required': True,
                    'schema': {'type': 'string'},
                }
            ],
        },
        {
            'name_for_human': '通讯录',
            'name_for_model': 'address_book',
            'description_for_model': '通讯录是用来获取个人信息如电话、邮箱地址、公司地址的软件。',
            'parameters': [
                {
                    'name': 'person_name',
                    'description': '被查询者的姓名',
                    'required': True,
                    'schema': {'type': 'string'},
                }
            ],
        },
        {
            'name_for_human': 'qq邮箱',
            'name_for_model': 'QQ_Email',
            'description_for_model': 'qq邮箱是一个可以用来发送合接受邮件的工具',
            'parameters': [
                {
                    'E-mail_address': 'E-mail_address',
                    'description': '对方邮箱的地址 发给对方的内容',
                    'required': True,
                    'schema': {'type': 'string'},
                },
                {'E-mail_content':"E-mail_content",
                 'description': '发给对方的内容',
                'required': True,
                'schema': {'type':'string'},}
            ],
        },
        {
            'name_for_human': '文生图',
            'name_for_model': 'image_gen',
            'description_for_model': '文生图是一个AI绘画（图像生成）服务，输入文本描述，返回根据文本作画得到的图片的URL',
            'parameters': [
                {
                    'name': 'prompt',
                    'description': '英文关键词，描述了希望图像具有什么内容',
                    'required': True,
                    'schema': {'type': 'string'},
                }
            ],
        },
        {
            'name_for_human': '态势显示',
            'name_for_model': 'Situation_display',
            'description_for_model': ':态势显示是通过输入目标位置坐标和显示范围，从而显示当前敌我双方的战场态势图像，并生成图片',
            'parameters': [
                {
                    'coordinate': '[coordinate_x,coordinate_y]',
                    'description': '目标位置的x和y坐标',
                    'required': True,
                    'schema': {'type': 'string'},
                }
                ,
                {
                    'radio': 'radio',
                    'description': '态势图像显示的范围半径,单位是km,默认值为300km',
                    'required': True,
                    'schema': {'type': 'string'},
                }
            ],
        },
        {
            'name_for_human': '万年历',
            'name_for_model': 'calendar',
            'description_for_model': '万年历获取当前时间的工具',
            'parameters': [
                {
                    'time': 'time_query',
                    'description':'目标的时间，例如昨天、今天、明天',
                    'location':'location_query',
                    'description': '目标的地点',
                    'required': True,
                    'schema': {'type': 'string'},
                }
            ],
        },
        {
            'name_for_human': '地图',
            'name_for_model': 'map_search',
            'description_for_model': '地图是一个可以查询地图上所有单位位置信息的工具，返回所有敌军的位置信息。',
            'parameters': [
                {
                    'lauch': 'yes',
                    'description': 'yes代表启用地图搜索',
                    'required': True,
                    'schema': {'type': 'string'},
                }
            ],
        },
        {
            'name_for_human': '知识图谱',
            'name_for_model': 'knowledge_graph',
            'description_for_model': '知识图谱是输入武器种类获取该武器的属性，也可以输入某种属性获得所有武器的该属性',
            'parameters': [
                {
                    'weapon': 'weapon_query',
                    'description': '武器名称,比如飞机、坦克,所有武器',
                    'required': True,
                    'schema': {'type': 'string'},
                },
                {
                    'attribute': 'attribute',
                    'description': '输出武器的该属性：射程/速度/重量/适应场景/克制武器/所有属性/续航里程/等等',
                    'required': True,
                    'schema': {'type': 'string'},
                }
            ],
        },
        {
            'name_for_human': 'python计算器',
            'name_for_model': 'python_math',
            'description_for_model': 'python计算器可以通过python的eval()函数计算出输入的字符串表达式结果并返回,表达式仅包含数字、加减乘除、逻辑运算符',
            'parameters': [
                {
                    'math_formulation': 'math_formulation',
                    'description': '根据问题提炼出的python数学表达式,表达式仅包含数字、加减乘除、逻辑运算符',
                    'required': True,
                    'schema': {'type': 'string'},
                }
            ],
        },
        {
            'name_for_human': '武器发射按钮',
            'name_for_model': 'weapon_launch',
            'description_for_model': '武器发射按钮是可以启动指定武器打击指定目标位置工具。',
            'parameters': [
                {
                    'weapon_and_coordinate': ('weapon_query','target_name',  ['x', 'y']),
                    'description': '被启动的武器名称 被打击的目标名称 被打击目标的坐标地点',
                    'required': True,
                    'schema': {'type': 'string'},
                }
            ],
        },
        {
            'name_for_human': '数学计算',
            'name_for_model': 'math_model',
            'description_for_model': '使用大语言模型完成一系列的推理问题如基本的加减乘除、最大、最小计算',
            'parameters': [
                {
                    'question': 'question',
                    'description': '当前的问题，需要清楚的给足背景知识',
                    'required': True,
                    'schema': {'type': 'string'},
                }
            ],
        },
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
        },
    ]

#以下是工具的实现部分，即调用工具后如何实现功能
def call_plugin(plugin_name: str, plugin_args: str,write_file,embeding_model=None,model=None,tokenizer=None,incontext=None,subtask=None):
    try:

        print("本次调用",plugin_name)
        print("本次参数",plugin_args)
        if plugin_name == 'google_search':
            # 使用 SerpAPI 需要在这里填入您的 SERPAPI_API_KEY！
            search_query=json5.loads(plugin_args)['search_query']
            #model.chat(tokenizer,task_split_prompt,history=[])
            return query_bing(search_query, max_tries=3,model=model,tokenizer=tokenizer)
        elif plugin_name == 'military_information_search':
            search_query=json5.loads(plugin_args)['search_query']
            return query_bing(search_query, max_tries=3,model=model,tokenizer=tokenizer)

            return SerpAPIWrapper().run(json5.loads(plugin_args)['search_query'])
        elif plugin_name == 'image_gen':
            import urllib.parse,json

            prompt = json5.loads(plugin_args)["prompt"]#输入的文本
            prompt = urllib.parse.quote(prompt)
            return json.dumps({'image_url': f'https://image.pollinations.ai/{prompt}'.format(prompt=prompt)}, ensure_ascii=False)
        elif plugin_name == 'QQ_Email':
            import urllib.parse,json
            Email_address = json5.loads(plugin_args)["E-mail_address"]
            Email_content = json5.loads(plugin_args)["E-mail_content"]
            return "已将{}发送到{}".format(Email_content,Email_address)
        elif plugin_name=='calendar':
            from datetime import datetime  
            format_str="%Y年%m月%d日 %H:%M:%S"
            # 如果提供了time_str，将字符串解析为datetime对象  
        
            time = datetime.now()  
            
            # 将时间格式化为字符串  
            formatted_time = time.strftime(format_str)  
        
            return formatted_time
        elif plugin_name=='math_model':
            question=json5.loads(plugin_args)['question']
            #result=model.chat(tokenizer,incontext,history=[])[0]
            #result=model.chat(tokenizer,incontext+'基于以上信息回答：'+question,history=[])[0]
            result=model.chat(tokenizer,incontext+'现在请你回答{}这个问题，并给出推理过程，最后给出答案。答案写成"Answer:答案"的格式'.format(subtask),history=[])[0]
            result=result.split('Answer:')[1]
            return result
        elif plugin_name=='map_search':#这个是地图信息的api
            map_dict={'我方直升机':[100,80],'敌直升机':[170,45],'我方指挥所':[0,2],'敌坦克':[20,13],'我方火箭炮':[100,120],'我方发射阵地1':[50,70],'我方发射阵地2':[150,170],"敌指挥中心": [70, 35],"敌反坦克导弹":[50,100],'我方坦克':[32,21]}
            import json
            args_dict = json.loads(plugin_args)
            #if args_dict['lauch']=='yes':
            return str(map_dict)
        elif plugin_name=='address_book':#这个是地图信息的api
            book_dict={'李四':{'邮箱':'403644786@qq.com','电话':'13077329411','部队':'直升机','职务':'算法工程师'},
                    '张三':{'邮箱':'45123456@qq.com','电话':'13713156111','部队':'黑狐坦克','职务':'副总师'},
                    '王五':{'邮箱':'45343438@qq.com','电话':'13745432','部队':'指挥所','职务':'C++开发'},
                    '我方指挥所':{'邮箱':'15sadf63@qq.com','电话':'062221234'},
                    '特种部队':{'邮箱':'112322233@qq.com','电话':'156123459','队长':'赵六'},
                    '侦察部队':{'邮箱':'1456412333@qq.com','电话':'056486123135','队长':'周八'},
                    '指挥官':{'邮箱':'123456789@qq.com','电话':'6220486123135'}
                    }
            import json
            args_dict = json.loads(plugin_args)
            person_name=args_dict['person_name']
            if person_name not in book_dict.keys():
                person_name=bm25(person_name,list(book_dict.keys()),model=embeding_model)#获取和query最相近的人名
            return str(book_dict[person_name])
        elif plugin_name=='knowledge_graph':#这个是知识图谱的api
            kg={'直升机':{'飞行高度':'0.3km以内','携带武器':'火箭弹','克制武器':'对空导弹','重量':'3000kg',"速度":"100km/h","射程":"2km",'适应场景':'空战','续航':'500km','满载人数':'7人','承载重量':'10000kg','续航里程':'1000km'},
                '反坦克导弹':{'重量':'100kg','射程':'0.5千米','克制武器':'拦截导弹','适应场景':'打击重装甲武器','速度':'200km/h'},
                '步兵':{'射程':'0.3km','克制武器':'无人机','适应场景':'陆地',"速度":'40km/h','重量':'60kg','承载重量':'50kg'},
                '无人机':{'速度':'100km/h','重量':'10kg','适应场景':'侦察和暗杀','飞行高度':'0.3km以下','克制武器':'电磁攻击','续航':'50km'},
                '豹2A7坦克':{'速度':'50km/h','携带武器':'激光炮','克制武器':'反坦克导弹',"射程":"5km",'重量':'10000kg','续航':'1000km','承载重量':'200000kg','满载人数':'5人' ,'适应场景' :'野战和掩护步兵'},
                '黑狐坦克':{'速度':'70km/h','携带武器':'主炮','克制武器':'反坦克导弹',"射程":"15km",'重量':'10000kg','承载重量':'50000kg','续航':'1000km','满载人数':'5人','适应场景' :'野战和掩护步兵'},
                "火箭炮":{'速度':'4500km/h','重量':'500kg','射程':'1000km','适应场景':'超远程打击','飞行高度':'万米高空','克制武器':'拦截导弹'},
                "雷达":{'重量':'5000kg','探测范围':'2km以上20km以下','适应场景':'探测敌军'},
                '装甲车':{'速度':'80km/h','携带武器':'副炮','克制武器':'穿甲弹',"射程":"0.5km",'重量':'10000kg','承载重量':'10000kg','续航':'600km','满载人数':'10人'},
                '狙击枪':{'射程':'1.2km','重量':'30kg','适应场景':'暗杀'}
                }
            import json
            args_dict=json.loads(plugin_args)
            if 'weapon_query' in args_dict.keys():#判断是否存在weapon参数
                weapon=args_dict['weapon_query']
                weapon=bm25(args_dict['weapon_query'],list(kg.keys())+['所有武器'],embeding_model)
            elif 'weapon' in args_dict.keys():#判断是否存在weapon参数
                weapon=args_dict['weapon']
                weapon=bm25(args_dict['weapon'],list(kg.keys())+['所有武器'],embeding_model)
            if 'attribute' in args_dict.keys() and args_dict['attribute']!='no':#如果属性存在
                    attribute=args_dict['attribute']
                    attribute=bm25(attribute,['射程','携带武器','重量','速度','适应场景','克制武器','续航','满载人数','飞行高度','承载重量','所有属性','探测范围'],embeding_model)
            #以上三行是获取属性和武器参数
            if 'weapon' in locals().keys() and 'attribute' in locals().keys():#有武器也有属性
                if weapon=='所有武器' and attribute!='所有属性':
                    weapon_string= ','.join([k+"的"+attribute+"是:"+v[attribute] for k,v in kg.items() if attribute in v.keys()])
                elif weapon!='所有武器' and attribute=='所有属性':
                    weapon_string= weapon+'的属性是：'+str(kg[weapon])
                elif weapon!='所有武器' and attribute!='所有属性':
                    if'/' in args_dict['attribute']:#如果属性中存在多个不同的query，如‘速度/飞行高度’
                        weapon_string=weapon+'的属性是：'+str(kg[weapon])
                    else:
                        try:
                            weapon_string= weapon+'的'+attribute+'是:'+str(kg[weapon][attribute])
                        except:
                            weapon_string=weapon+'的属性是：'+str(kg[weapon])
            elif "weapon" in locals().keys() and "attribute" not in locals().keys():#有武器没有属性
                if weapon!='所有武器':
                    weapon_string= weapon+'的属性是：'+str(kg[weapon])
            elif "weapon" not in locals().keys() and "attribute" in locals().keys():#没有武器有属性
                if attribute!='所有属性':
                    weapon_string= ','.join([k+"的"+attribute+"是:"+v[attribute] for k,v in kg.items() if attribute in v.keys()])
            return weapon_string
        elif plugin_name=='weapon_launch':#这里是发射武器的api
            import json
            args_dict = json.loads(plugin_args)
            weapon=args_dict["weapon_and_coordinate"][0]
            target_name = args_dict["weapon_and_coordinate"][1]
            coordinate=args_dict["weapon_and_coordinate"][2]
            return '已启动'+weapon+'打击'+str(target_name)+'打击位置：'+str(coordinate)
        elif plugin_name=='Situation_display':#这里是发射武器的api
            import json
            args_dict = json.loads(plugin_args)
            if 'radio' not in args_dict.keys():
                radio=300
            else:
                radio=args_dict["radio"]
            coordinate = args_dict["coordinate"]
            return '已经显示以{}为中心以{}为半径的态势地图,图片地址为/ai/ld/picture1'.format(coordinate,radio)
        elif plugin_name=='distance_calculation':#这里是计算距离的api
            import json,ast
            args_dict = json.loads(plugin_args)
            weapon=list(args_dict["target_and_mapdict"].keys())[0]#传进来的武器参数
            coordinate=args_dict["target_and_mapdict"]#传进来的整个位置信息
            items=list(coordinate.keys())
            query=bm25(weapon,items,embeding_model)
            distance_list=distance(query,coordinate)
            min_distance_unit=min(distance_list[1],key=distance_list[1].get)
            max_distance_unit=max(distance_list[1],key=distance_list[1].get)
            return '以下是所有单位与{}的距离:{}'.format(query,str(distance_list[1]))
        elif plugin_name=='python_math':#这里是计算距离的api
            import json
            args_dict = json.loads(plugin_args)
            math_formulation=args_dict['math_formulation']
            
            math_formulation=re.sub('km','',math_formulation)
            math_formulation=re.sub('km/h','',math_formulation)
            try:
                result=eval(math_formulation)

                return "执行结果是{}".format(str(result))
            except:
                pattern = re.compile(r"[^0-9/+*-/<>=%()](max|min|abs|pow|sqrt|exp)")
                math_formulation=re.sub(pattern,'',math_formulation)
                try:
                    result=eval(math_formulation)

                    return "执行结果是{}".format(str(result))
                except:
                    return '执行失败'
        else:
            return '没有找到该工具'
    except:
        return '执行失败'