
## Ollama

1. **获取我们fork的分支代码**
   PR暂未合并，请务必使用我们的分支
   ```bash
   git clone https://github.com/LDLINGLINGLING/ollama.git
   cd ollama
   git checkout minicpm
   tar -zxvf ollama.tar.gz -C .
   cd ollama
   ```

2. **编译Ollama**
   ```bash
   cd ollama
   go generate ./... # 这个过程需要访问google等网址
   go build .
   ```

3. **按照[Llamacpp的教程](llamcpp.md)获取量化后的gguf文件**

4. **在Ollama主路径下开启服务**
   ```bash
   ./ollama serve
   ```

5. **创建一个Model.file文件**
   ```bash
   vim minicpm3.ModelFile
   ```
   文件内容如下：
   ```plaintext
    FROM ./MiniCPM-V-2_6/ggml-model-Q4_K_M.gguf
    TEMPLATE """{{ if .System }}<|im_start|>system
    {{ .System }}<|im_end|>{{ end }}
    {{ if .Prompt }}<|im_start|>user
    {{ .Prompt }}<|im_end|>{{ end }}
    <|im_start|>assistant<|im_end|>
    {{ .Response }}<|im_end|>"""

    PARAMETER stop "<|endoftext|>"
    PARAMETER stop "<|im_end|>"
    PARAMETER num_ctx 4096
    """
   ```

6. **创建Ollama实例**
   ```bash
   ollama create minicpm -f minicpm3.ModelFile
   ```

7. **运行模型**
   ```bash
   ollama run minicpm3
   ```
