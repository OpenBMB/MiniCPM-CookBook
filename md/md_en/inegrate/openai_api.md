
# OpenAI API
## Usage
[Project Repository](https://github.com/OpenBMB/MiniCPM/tree/main/demo/openai_api_demo)

1. First, obtain the official MiniCPM code:
   ```sh
   git clone https://github.com/OpenBMB/MiniCPM.git
   cd MiniCPM/demo/openai_api_demo
   ```

2. Modify the following parameters in `MiniCPM/demo/openai_api_demo/openai_api_server_demo.py` as needed:
   ```python
   MODEL_PATH = os.environ.get('MODEL_PATH', 'openbmb/MiniCPM-2B-dpo-fp16')
   TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)
   EMBEDDING_PATH = os.environ.get('EMBEDDING_PATH', 'BAAI/bge-m3')
   ```

3. Run the `MiniCPM/demo/openai_api_demo/openai_api_server_demo.py` file:
   ```sh
   python openai_api_server_demo.py
   ```

4. Modify the request content in the `MiniCPM/demo/openai_api_demo/openai_api_request_demo.py` file:
   ```python
   def chat(use_stream=True):
       messages = [
           {
               "role": "user",
               "content": "tell me a story"  # If the request does not carry historical context, only modify the content
           }
       ]
       # To carry historical context, use the following format
       # messages = [
       #     {
       #         "role": "user",
       #         "content": "hello"
       #     },
       #     {
       #         "role": "assistant",
       #         "content": "hello, can i help you?"
       #     },
       #     {
       #         "role": "user",
       #         "content": "can you tell me a story?"
       #     }
       # ]
   ```

5. After making the modifications, run `MiniCPM/demo/openai_api_demo/openai_api_request_demo.py` to receive the response:
   ```sh
   python openai_api_request_demo.py
   ```