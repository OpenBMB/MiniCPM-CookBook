import os
import json
import random
# 获取mbti_sft_dpo_data目录下的所有文件和文件夹名称
path='MIniCPM_Series_Tutorial/mbti_role_play/mbti_sft_dpo_data'
out_put_dir='MIniCPM_Series_Tutorial/mbti_role_play/out_data'
entries = os.listdir(path)
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# 过滤出文件，排除文件夹
files = [entry for entry in entries if entry.endswith('.json')]
motion_dict1={'E':'energy_extraversion','N':'information_intuition','T':'decision_thinking','J':'execution_judging'}
motion_dict2={'I':'energy_introversion','S':'information_sensing','F':'decision_feeling','P':'execution_perceiving'}

decision_feeling_file=load_json(os.path.join(path,'zh_decision_feeling.json'))
decision_thinking_file=load_json(os.path.join(path,'zh_decision_thinking.json'))
energy_introversion_file=load_json(os.path.join(path,'zh_energy_introversion.json'))
energy_extraversion_file=load_json(os.path.join(path,'zh_energy_extraversion.json'))
execution_judging_file=load_json(os.path.join(path,'zh_execution_judging.json'))
execution_perceiving_file=load_json(os.path.join(path,'zh_execution_perceiving.json'))
information_intuition_file=load_json(os.path.join(path,'zh_information_intuition.json'))
information_sensing_file=load_json(os.path.join(path,'zh_information_sensing.json'))

all_rank_data=[]
all_sft_data=[]
for index,type1 in enumerate(motion_dict1.keys()):
    temp_rank_data=[]
    good_data=load_json(os.path.join(path,'zh_')+motion_dict1[type1]+'.json')
    bad_data=load_json(os.path.join(path,'zh_')+list(motion_dict2.values())[index]+'.json')
    for ind,i in enumerate(good_data):
        random_numbers = [random.choice([0, 1]) for _ in range(3)]
        random_numbers.insert(index,0)
        type_list_temp=[]
        for i_1 in range(4):
            if i_1==index:
                type_list_temp.append(type1)
            else:
                if random_numbers[i_1]==0:
                    type_list_temp.append(list(motion_dict1.keys())[i_1])
                else:
                    type_list_temp.append(list(motion_dict2.keys())[i_1])
        print(type_list_temp)
        prompt='<SEP>{}<SEP>\n'.format(''.join(type_list_temp))
        i['instruction']=prompt+i['instruction']
        line_message={"conversations": [{"from": "human","value": i['instruction']}],"chosen": {"from": "gpt","value":i['output']},"rejected": {"from": "gpt","value":bad_data[ind]['output']}}
        temp_rank_data.append(line_message)
        all_sft_data.append(i)
    all_rank_data+=temp_rank_data

for index,type1 in enumerate(motion_dict2.keys()):
    temp_rank_data=[]
    random_numbers = [random.choice([0, 1]) for _ in range(3)]
    good_data=load_json(os.path.join(path,'zh_')+motion_dict2[type1]+'.json')
    bad_data=load_json(os.path.join(path,'zh_')+list(motion_dict1.values())[index]+'.json')
    for ind,i in enumerate(good_data):
        random_numbers = [random.choice([0, 1]) for _ in range(3)]
        random_numbers.insert(index,1)
        type_list_temp=[]
        for i_1 in range(4):
            if i_1==index:
                type_list_temp.append(type1)
            else:
                if random_numbers[i_1]==0:
                    type_list_temp.append(list(motion_dict1.keys())[i_1])
                else:
                    type_list_temp.append(list(motion_dict2.keys())[i_1])
        print(type_list_temp)
        prompt='<SEP>{}<SEP>\n'.format(''.join(type_list_temp))
        i['instruction']=prompt+i['instruction']
        line_message={"conversations": [{"from": "human","value": str(i['instruction'])}],"chosen": {"from": "gpt","value":str(i['output'])},"rejected": {"from": "gpt","value":bad_data[ind]['output']}}
        temp_rank_data.append(line_message)
        all_sft_data.append(i)
    all_rank_data+=temp_rank_data
with open(os.path.join(out_put_dir,'mbti_rank.json'), 'w', encoding='utf-8') as file:
    json.dump(all_rank_data, file, ensure_ascii=False, indent=4)
with open(os.path.join(out_put_dir,'mbti_sft.json'), 'w', encoding='utf-8') as file:
    json.dump(all_sft_data, file, ensure_ascii=False, indent=4)

