# LLaMA-Factory Usage Guide

The usage of LLaMA-Factory is largely similar to the [Fine-Tuning Tutorial for MiniCPM2.0](../minicpm2.0/llama_factory.md), with the exception of changing `template: cpm3`.

1. **Install LLaMA-Factory Dependencies**
   My personal environment setup can be found in the MiniCPM3 [pip_list](pip_list.md).
   ```bash
   git clone https://github.com/hiyouga/LLaMA-Factory.git
   cd LLaMA-Factory
   pip install -e ".[torch,metrics]"
   ```

2. **Data Processing**
   Process the dataset into the format found in the `Minicpm/finetune/llama_factory_example/llama_factory_data` folder, including examples for three fine-tuning methods: DPO, KTO, and SFT, and place them under the `llama_factory/data` directory.

   #### 2.1 DPO Data Format
     For a "human" question, provide a chosen (good answer) and a rejected (bad answer). The parts that need modification are highlighted.
     ```json
     [
       {
         "conversations": [
           {
             "from": "human",
             "value": "Have you eaten? What did you eat?"
           },
           {
             "from": "gpt",
             "value": "Just ate, where are you going?"
           },
           {
             "from": "gpt",
             "value": "Had eggplant, tomatoes, and braised pork."
           }
         ],
         "chosen": {
           "from": "gpt",
           "value": "Just ate, where are you going?"
         },
         "rejected": {
           "from": "gpt",
           "value": "Had eggplant, tomatoes, and braised pork."
         }
       }
     ]
     ```

   #### 2.2 KTO Data Format
     1. Each `messages` represents a piece of data, and `label` indicates whether the response to that piece of data is reasonable.
     2. Unlike DPO, you don't need different answers to the same question; instead, evaluate each question-answer pair.
     3. Each question and response can receive a True (good) or False (bad) rating.
     4. The parts that need modification are highlighted.
     ```json
     [
       {
         "messages": [
           {
             "content": "Where is MiniCPM3.0 the strongest?",
             "role": "user"
           },
           {
             "content": "It's strong everywhere, isn't it? A hexagonal warrior.",
             "role": "assistant"
           }
         ],
         "label": true
       },
       {
         "messages": [
           {
             "content": "What are the advantages of MiniCPM3.0?",
             "role": "user"
           },
           {
             "content": "1. Long text capability. 2. Function call capability 3. All other capabilities",
             "role": "assistant"
           },
           {
             "content": "Explain the long text capability in detail.",
             "role": "user"
           },
           {
             "content": "It's very strong, go to our GitHub for details. https://github.com/OpenBMB/MiniCPM.git",
             "role": "assistant"
           }
         ],
         "label": false
       }
     ]
     ```

   #### 2.3 SFT Data Format, Each Dictionary Forms a Piece of Data
     ```json
     [
       {
         "instruction": "You are now a scientist even more brilliant than Newton.",
         "input": "Identify and explain the two scientific theories given in the list: cell theory and heliocentrism.",
         "output": "Cell theory is a theory in biological science that posits all living organisms are composed of tiny fundamental units called cells. This is a foundational theory in biology, suggesting that cells are the basic structural and functional units of all living things, and that all living beings are made up of one or more cells, which can only arise through cell division. This theory was first proposed by Schleiden, Schwann, and Virchow in 1839.\n\nHeliocentrism refers to the theory that the Sun is at the center of the solar system, meaning that planets revolve around the Sun. This theory challenged the traditional geocentric view, which held that Earth was the center of the universe. The proponent of heliocentrism was Nicolaus Copernicus, who published his work 'De revolutionibus orbium coelestium' (On the Revolutions of the Celestial Spheres) in the early 16th century, outlining the model of planets orbiting the Sun, making a significant contribution to the development of astronomy."
       }
     ]
     ```

3. **Add Data Information to `dataset_info.json`**
   Add dataset information in `llama_factory/data/dataset_info.json` to ensure that your dataset can be found in `dataset_info.json`. Example:
   ```json
   {
     "identity": {
       "file_name": "identity.json"
     },
     "sft_zh_demo": {
       "file_name": "alpaca_zh_demo.json"
     },
     "kto_en_demo": {
       "file_name": "kto_en_demo.json",
       "formatting": "sharegpt",
       "columns": {
         "messages": "messages",
         "kto_tag": "label"
       },
       "tags": {
         "role_tag": "role",
         "content_tag": "content",
         "user_tag": "user",
         "assistant_tag": "assistant"
       }
     },
     "dpo_en_demo": {
       "file_name": "dpo_en_demo.json",
       "ranking": true,
       "formatting": "sharegpt",
       "columns": {
         "messages": "conversations",
         "chosen": "chosen",
         "rejected": "rejected"
       }
     }
   }
   ```

4. **Set Up Training Scripts**

   #### 4.1 Copy Files from `MiniCPM/finetune/llama_factory_example` to `LLaMA-Factory/examples/minicpm` Directory
   ```bash
   cd LLaMA-Factory/examples
   mkdir minicpm
   # Replace /your/path with the actual paths to your MiniCPM and LLaMA-Factory directories
   cp -r /your/path/MiniCPM/finetune/llama_factory_example/* /your/path/LLaMA-Factory/examples/minicpm
   ```

   #### 4.2 Modify Configuration Parameters in `LLaMA-Factory/examples/minicpm/minicpm_dpo.yaml` Based on Your Fine-Tuning Method, Using DPO as an Example
   ```yaml
   model_name_or_path: openbmb/MiniCPM3-4B # Or the path where you have saved the model locally
   template: cpm3 # Note: If fine-tuning MiniCPM3, write cpm3; otherwise, write cpm
   dataset: dpo_en_demo # Write the key name from dataset_info.json here
   output_dir: your/finetune_minicpm/save/path # The location where your fine-tuned model will be saved
   bf16: true # If your device supports bf16, otherwise set to false
   deepspeed: examples/deepspeed/ds_z2_config.json # If GPU memory is insufficient, change to ds_z3_config.json
   ```

   #### 4.3 Modify the Following Configurations in `LLaMA-Factory/examples/minicpm/single_node.sh` File
   ```bash
   # Modify the following line to reflect the number of GPUs on your machine
   NPROC_PER_NODE=8
   NNODES=1
   RANK=0
   MASTER_ADDR=127.0.0.1
   MASTER_PORT=29500

   # The following two lines can be deleted if you have high-end GPUs such as A100, H100, etc.
   export NCCL_P2P_DISABLE=1
   export NCCL_IB_DISABLE=1 

   # Set the following numbers to the GPUs participating in training on your machine, here GPUs 0-7 are all participating in training
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
     --nproc_per_node $NPROC_PER_NODE \
     --nnodes $NNODES \
     --node_rank $RANK \
     --master_addr $MASTER_ADDR \
     --master_port $MASTER_PORT \

   # Change the following line to the path of the configuration file, e.g., /root/ld/ld_project/LLaMAFactory/examples/minicpm/minicpm_dpo.yaml
     src/train.py /your/path/LLaMA-Factory/examples/minicpm/minicpm_dpo.yaml
   ```

5. **Start Training**
   ```bash
   cd LLaMA-Factory
   bash /your/path/LLaMA-Factory/examples/minicpm/single_node.sh
   ```
