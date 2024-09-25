# Ollama Deployment

## System Requirements
- Running non-quantized version: More than 19GB of memory
- Running quantized version: More than 8GB of memory

## Step 1: Obtain the gguf Model

Follow the [llama.cpp tutorial](llamacpp_pc.md) to obtain the gguf model file, and it is recommended to perform quantization on the language model.

## Step 2: Get the Official Ollama Branch from OpenBMB

Clone the specified branch using Git:

```sh
git clone -b minicpm-v2.5 https://github.com/OpenBMB/ollama.git
cd ollama/llm
```

## Step 3: Ensure Environment Dependencies

Ensure the following dependencies are met:
- CMake version 3.24 or higher
- Go version 1.22 or higher
- GCC version 11.4.0 or higher

Install these dependencies using Homebrew:

```sh
brew install go cmake gcc
```

## Step 4: Install Large Model Dependencies

Install the large model dependencies for Ollama:

```sh
go generate ./...
```

## Step 5: Compile Ollama

Compile Ollama:

```sh
go build .
```

## Step 6: Start the Ollama Service

After successful compilation, start the service in the Ollama root directory:

```sh
./ollama serve
```

## Step 7: Create the Model File

Create a file named `minicpmv2_5.Modelfile`:

```sh
vim minicpmv2_5.Modelfile
```

The content of the file should be:

```plaintext
# Replace the paths after the first and second FROM with the paths to the quantized language model and the image projection model, respectively.

FROM ./MiniCPM-V-2_5/model/ggml-model-Q4_K_M.gguf
FROM ./MiniCPM-V-2_5/mmproj-model-f16.gguf

TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"""

PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
PARAMETER num_keep 4
PARAMETER num_ctx 2048
```

## Step 8: Create the Ollama Model

Create the Ollama model using the following command:

```sh
ollama create minicpm2.5 -f minicpmv2_5.Modelfile
```

## Step 9: Run the Ollama Model

Run the created Ollama model:

```sh
ollama run minicpm2.5
```

## Step 10: Input Questions and Image Paths

When inputting questions and image paths, use spaces to separate them.

Now you are ready to start efficient inference with Ollama!
