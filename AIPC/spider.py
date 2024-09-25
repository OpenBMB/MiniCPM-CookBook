import requests
from bs4 import BeautifulSoup
import json
# 目标URL
url = 'https://liubing.me/article/mac/mac-shortcut-keys.html#复制'

# 发送HTTP请求获取页面内容
response = requests.get(url)

# 检查请求是否成功
if response.status_code == 200:
    # 获取网页的编码方式
    response.encoding = response.apparent_encoding
    
    # 解析HTML文档
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # 找到指定路径下的div标签
    div_tag = soup.select_one('#main-content > div:nth-of-type(3)')
    feature = []
    Shortcut = []
    if div_tag:
        # 获取div标签下的一级子标签中的h标签和p标签
        first_level_h_tags = div_tag.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'], recursive=False)
        first_level_p_tags = div_tag.find_all('p', recursive=False)
        
        print(len(first_level_h_tags) == len(first_level_p_tags))
        print("找到了指定路径下的所有一级子标签中的h标签：")
        for tag in first_level_h_tags[2:-1]:
            print(tag.name)
            print(tag.text.strip())
            feature.append(tag.text.strip())
        
        print("\n找到了指定路径下的所有一级子标签中的p标签：")
        for tag in first_level_p_tags[1:]:
            print(tag.name)
            print(tag.text.strip())
            Shortcut.append(tag.text.strip())
    
    else:
        print("没有找到指定路径下的div标签")
else:
    print(f"请求失败，状态码: {response.status_code}")
dictionary = {key: value for key, value in zip(feature , Shortcut)}
with open('/Users/liudan/ai/MiniCPM_Series_Tutorial/AIPC/Mac_feature_and_shortcut.json', 'w') as json_file:
    json.dump(dictionary, json_file, indent=4,ensure_ascii=False)