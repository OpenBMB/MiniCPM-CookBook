# openai_api
## 使用方法
[项目地址](https://github.com/OpenBMB/MiniCPM/tree/main/demo/openai_api_demo)
1. 首先获取MiniCPM官方代码
```sh
git clone https://github.com/OpenBMB/MiniCPM.git
cd MiniCPM/demo/openai_api_demo
```
2. 根据需要修改`MiniCPM/demo/openai_api_demo\openai_api_server_demo.py`以下参数
```python
MODEL_PATH = os.environ.get('MODEL_PATH', 'openbmb/MiniCPM-2B-dpo-fp16')
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)
EMBEDDING_PATH = os.environ.get('EMBEDDING_PATH', 'BAAI/bge-m3')
```
3. 运行`MiniCPM/demo/openai_api_demo\openai_api_server_demo.py`文件：
```sh
python openai_api_server_demo.py
```
4. 修改`MiniCPM/demo/openai_api_demo/openai_api_request_demo.py`文件中的请求内容。
```python
def chat(use_stream=True):
    messages = [
        {
            "role": "user",
            "content": "tell me a story" # 如果不携带历史上下问的请求，可以仅修改content
        }
    ]
    # 携带历史上下问如下
    # messages = [
    #     {
    #         "role": "user",
    #         "content": "hello" 
    #     },
    #     {
    #       "role": "assistant",
    #       "content": "hello, can i help you?" 
    #      }
    #     {
    #       "role": "user",
    #       "content": "can you tell me a story?" 
    #      }
    # ]
```
5. 修改后，直接运行`MiniCPM/demo/openai_api_demo/openai_api_request_demo.py`即可以获得回复。