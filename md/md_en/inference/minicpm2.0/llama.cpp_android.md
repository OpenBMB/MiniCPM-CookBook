# Deploying llama.cpp on Android

## Device Requirements
- Android smartphone
- It is recommended to use a phone with Snapdragon 8 series or better chipset.

### Step 1: Obtain the ggml-model-Q4_K_M.gguf Quantized Model

Follow the instructions in [Deploying llama.cpp on PC](llama.cpp_pc.md) to acquire and quantize the model file `ggml-model-Q4_K_M.gguf`.

### Step 2: Transfer the Model File to Your Phone

Transfer the quantized model file to the `/sdcard/Download` directory on your phone. Here, we provide a method using ADB (Android Debug Bridge), although other methods can also be used:

```sh
adb push ggml-model-Q4_K_M.gguf /sdcard/Download
```

### Step 3: Install Termux

Download and install an appropriate version of Termux on your phone; it is recommended to use [version v0.118.1](https://github.com/termux/termux-app/releases/tag/v0.118.1).


### Step 4: Grant Storage Access Permissions to Termux

Open the Termux application and run the following command to grant storage permissions to Termux:

```sh
termux-setup-storage
```

### Step 5: Fetch and Compile the llama.cpp Source Code

Fetch the llama.cpp source code and compile it within Termux:

```sh
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make main
```

### Step 6: Perform Inference on the Mobile Device

Use the compiled llama-cli tool to perform inference:

```sh
./llama-cli -m /sdcard/Download/ggml-model-Q4_K_M.gguf --prompt "<User>Do you know openmbmb?<AI>"
```

Now you can start performing inference using the MiniCPM model on your Android device!

---
Please note that some of the commands mentioned above may need to be adjusted according to your specific environment, such as the version number of Termux or other details.
