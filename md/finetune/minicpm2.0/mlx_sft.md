

# 使用MLX训练指南（Mac推荐）

## 设备要求
- Mac OS 14以上版本

## 步骤
1. **下载LLAMA_FORMAT_MLX中的所有文件**
   将所有文件下载到 `model_path` （路径可自定义）。

2. **构造数据**
   
   ### 方法1: 使用Demo数据
   - 运行 `MiniCPM/finetune/data_processing.ipynb` 进行数据处理。
   - 获得 `MiniCPM/finetune/data/mlx_AdvertiseGen` 路径下的 `train.json` 和 `dev.json` 文件。

   ### 方法2: 自定义数据
   - 数据集目录存放 `train.json` 和 `dev.json` 的训练数据和测试数据：
     ```
     -data_path
         -train.json
         -dev.json
     ```
   - JSON数据保持以下格式，每行一个字典，每个字典是一条数据，字典存在 `prompt`, `input`, `output` 三个键：
     ```json
     {"input": "类型#裙*材质#蕾丝*风格#宫廷*图案#刺绣*图案#蕾丝*裙型#大裙摆*裙下摆#花边*裙袖型#泡泡袖", "prompt": "\n请为以下关键词生成一条广告语。", "output": "宫廷风的甜美蕾丝设计，清醒的蕾丝拼缝处，刺绣定制的贝壳花边，增添了裙子的精致感觉。超大的裙摆，加上精细的小花边设计，上身后既带着仙气撩人又很有女人味。泡泡袖上的提花面料，在细节处增加了浪漫感，春日的仙女姐姐。浪漫蕾丝布满整个裙身，美丽明艳，气质超仙。"}
     {"input": "类型#裤*版型#显瘦*颜色#黑色*风格#简约*裤长#九分裤", "prompt": "\n请为以下关键词生成一条广告语。", "output": "个性化的九分裤型，穿着在身上，能够从视觉上拉长你的身体比例，让你看起来更加的有范。简约的黑色系列，极具时尚的韵味，充分凸显你专属的成熟韵味。修身的立体廓形，为你塑造修长的曲线。"}
     {"input": "类型#裙*版型#显瘦*风格#文艺*风格#简约*图案#印花*图案#撞色*裙下摆#压褶*裙长#连衣裙*裙领型#圆领", "prompt": "\n请为以下关键词生成一条广告语。", "output": "文艺个性的印花连衣裙，藏青色底蕴，低调又大气，撞色太阳花分布整个裙身，绚丽而美好，带来时尚减龄的气质。基础款的舒适圆领，简约不失大方，勾勒精致脸庞。领后是一粒包布扣固定，穿脱十分方便。前片立体的打褶设计，搭配后片压褶的做工，增添层次和空间感，显瘦又有型。"}
     ```

3. **处理数据后开始训练**
   ```sh
   cd MiniCPM/finetune
   python mlx_finetune.py --model model_path  --data data/mlx_AdvertiseGen  --train  --seed 2024 --iters 500
   ```

4. **找到 `adapters.npz` 文件并开启测试**
   ```sh
   python mlx_finetune.py --model model_path  --data data/mlx_AdvertiseGen  --test --seed 2024 --adapter-file adapters.npz
   ```

5. **使用以下代码进行推理**
   ```sh
   python mlx_finetune.py --model model_path  --seed 2024 --prompt '<用户>#拉链*裙款式#吊带*裙款式#收腰\n请为以上关键词生成一句广告语。<AI>' --adapter-file adapters.npz
   ```
