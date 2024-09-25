## Setting Up the Environment

1. Clone the MiniCPM repository by entering the following command in the terminal:
   ```bash
   git clone https://github.com/OpenBMB/MiniCPM.git
   ```

2. Install the required dependencies for the project:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the `MiniCPM/demo/hf_demo.py` file and check the following parameter settings:
   ```python
   parser = argparse.ArgumentParser()
   parser.add_argument("--model_path", type=str, default="openbmb/MiniCPM-2B-dpo-fp16")
   parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])
   parser.add_argument("--server_name", type=str, default="127.0.0.1")
   parser.add_argument("--server_port", type=int, default=7860)
   args = parser.parse_args()
   ```

4. If you have downloaded the model and wish to change the default path, modify the following parameters according to your actual situation:
   ```diff
   - parser.add_argument("--model_path", type=str, default="openbmb/MiniCPM-2B-dpo-fp16")
   + parser.add_argument("--model_path", type=str, default="/root/ai/minicpm_model")
   ```

   If your GPU is not a high-end model like the A100, 4090, or H100, it is recommended to set the precision to `float16`:
   ```diff
   - parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])
   + parser.add_argument("--torch_dtype", type=str, default="float16", choices=["float32", "bfloat16", "float16"])
   ```

5. Finally, run the `hf_demo.py` script:
   ```bash
   python MiniCPM/demo/hf_demo.py
   ```
   ![Image](../../../../asset/image.png)

6. Enter the provided address in your web browser, or click to open it.

7. Enjoy using MiniCPM!
