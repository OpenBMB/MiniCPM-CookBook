import requests
import json
import base64
with open(r'D:\model_best\minicpm\infer\images\1564165156.jpg', 'rb') as image_file:
        # 将图片文件转换为 base64 编码
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
data = {
"model": "minicpm-v",
"prompt": "最近半个月是涨还是跌",
"stream": False,
"images": [encoded_string]
}

# 设置请求 URL
url = "http://localhost:11434/api/generate"
response = requests.post(url, json=data)

print( response.json()["response"])