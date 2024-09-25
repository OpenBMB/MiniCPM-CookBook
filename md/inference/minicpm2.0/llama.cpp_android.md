# 部署llama.cpp到安卓端

## 设备要求
- 安卓手机
- 推荐使用骁龙8系列及以上芯片的手机

### 步骤1：获得ggml-model-Q4_K_M.gguf量化模型

按照[部署llama.cpp到PC端](llama.cpp_pc.md)获取并量化模型文件`ggml-model-Q4_K_M.gguf`。

### 步骤2：将模型文件传输到手机

将量化后的模型文件传输到手机的`/sdcard/Download`目录中。这里提供一种使用ADB（Android Debug Bridge）的方法，当然也可以使用其他方式：

```sh
adb push ggml-model-Q4_K_M.gguf /sdcard/Download
```

### 步骤3：安装Termux

在手机上下载并安装合适的Termux版本，推荐使用[v0.118.1版本](https://github.com/termux/termux-app/releases/tag/v0.118.1)。

### 步骤4：授权Termux访问存储权限

打开Termux应用，并运行以下命令以授予Termux存储权限：

```sh
termux-setup-storage
```

### 步骤5：获取并编译llama.cpp源码

在Termux中获取llama.cpp的源码并进行编译：

```sh
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make main
```

### 步骤6：在手机端进行推理

使用编译好的llama-cli工具进行推理：

```sh
./llama-cli -m /sdcard/Download/ggml-model-Q4_K_M.gguf --prompt "<用户>你知道openmbmb么<AI>"
```

现在您可以开始在安卓设备上使用MiniCPM模型进行推理了！

---
请注意，上述步骤中的某些命令可能需要根据您的具体环境进行调整，例如Termux的版本号或其他细节。

