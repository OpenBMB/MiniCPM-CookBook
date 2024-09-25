import os
import json

# 写入MIniCPM_Series_Tutorial/mbti_role_play/self_awareness的绝对地址
directory ='/root/ld/ld_project/mbti_role_play/self_awareness'
# 使用os.listdir()获取目录下的所有文件和子目录名
filenames = os.listdir(directory)

all_data=[]
# 打印文件名
for filename in filenames:
    if not filename.endswith('.json'):
        continue
    # 使用with语句确保文件正确关闭
    with open(os.path.join(directory,filename), 'r', encoding='utf-8') as file:
        # 将JSON数据加载到data变量中
        print(filename)
        data = json.load(file)
        all_data.extend(data)

with open(os.path.join(directory,'all_self_awarness.json'), 'w', encoding='utf-8') as file:
    json.dump(all_data, file, ensure_ascii=False, indent=4)