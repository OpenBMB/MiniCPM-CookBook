# Installing LLaMA-Factory Dependencies

First, clone the LLaMA-Factory repository and install the dependencies:
```sh
git clone https://github.com/hiyouga/LLaMA-Factory
cd LLaMA-Factory
pip install -r requirements.txt
```

# Data Processing

Process the dataset into the format found in the `Minicpm/finetune/llama_factory_example/llama_factory_data` folder and place it under the `llama_factory/data` directory. Examples include three fine-tuning methods: DPO, KTO, and SFT.

### 2.1 DPO Data Format
```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "Hi! I'd like to create a new language game simulating the first person perspective of a character named Angela."
      }
    ],
    "chosen": {
      "from": "gpt",
      "value": "That sounds like a fun and engaging idea! Here are some tips to help you create the game:\n1. Start with the character's name and background: "
    },
    "rejected": {
      "from": "gpt",
      "value": "Hello! 😊"
    }
  }
]
```

### 2.2 KTO Data Format
Each `messages` represents a piece of data, and `label` indicates whether the response to that piece of data is reasonable.
```json
{
  "messages": [
    {
      "content": "The Federal Trade Commission is going after spyware, bits of computer code that install themselves on the computers of Internet users to track or disrupt their online activities.\nQ: Which is the best summary of this article?\nPick your answer from:\n(A). World\n(B). Sports\n(C). Business\n(D). Science/Tech\nI think the answer is",
      "role": "user"
    },
    {
      "content": "Ooh ooh ah ah! Science/Tech! Yay! *confetti* The Federal Trade Commission is talking about spyware, which is like a little computer bug that can watch what you do on your computer without you knowing. It's like a secret agent, but not a nice one! *gasp* The FTC wants to stop the spyware from doing its thing, so they're going after it! *cheer* Woohoo!",
      "role": "assistant"
    }
  ],
  "label": true // true or false, true is a good response, false is a bad response
}
```

### 2.3 SFT Data Format
Each dictionary forms a piece of data.
```json
[
  {
    "instruction": "Identify and explain the two scientific theories given in the list: cell theory and heliocentrism.",
    "input": "",
    "output": "Cell theory is a theory in biological science that posits all living organisms are composed of tiny fundamental units called cells. This is a foundational theory in biology, suggesting that cells are the basic structural and functional units of all living things, and that all living beings are made up of one or more cells, which can only arise through cell division. This theory was first proposed by Schleiden, Schwann, and Virchow in 1839.\n\nHeliocentrism refers to the theory that the Sun is at the center of the solar system, meaning that planets revolve around the Sun. This theory challenged the traditional geocentric view, which held that Earth was the center of the universe. The proponent of heliocentrism was Nicolaus Copernicus, who published his work 'De revolutionibus orbium coelestium' (On the Revolutions of the Celestial Spheres) in the early 16th century, outlining the model of planets orbiting the Sun, making a significant contribution to the development of astronomy."
  }
]
```

# Add Data Information to `dataset_info.json`

Add dataset information in `llama_factory/data/dataset_info.json` to ensure that your dataset can be found in `dataset_info.json`.
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

# Setting Up Training Scripts

### 4.1 Copy Example Files
Copy the files in `MiniCPM/finetune/llama_factory_example` to the `LLaMA-Factory/examples/minicpm` directory.
```sh
cd LLaMA-Factory/examples
mkdir minicpm
cp -r /your/path/MiniCPM/finetune/llama_factory_example/* /your/path/LLaMA-Factory/examples/minicpm
```

### 4.2 Modify Configuration File
Based on the fine-tuning method you choose, using DPO as an example. You must modify the configuration parameters in `LLaMA-Factory/examples/minicpm/minicpm_dpo.yaml` as follows:
```yaml
model_name_or_path: openbmb/MiniCPM-2B-sft-bf16 # Or the path where you have saved the model locally
dataset: dpo_en_demo # Write the key name from dataset_info.json here
output_dir: your/finetune_minicpm/save/path # The location where your fine-tuned model will be saved
bf16: true # If your device supports bf16, otherwise set to false
deepspeed: examples/deepspeed/ds_z2_config.json # If GPU memory is insufficient, change to ds_z3_config.json
```

### 4.3 Modify `single_node.sh` File
Modify the following configurations in the `LLaMA-Factory/examples/minicpm/single_node.sh` file:
```sh
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

# Change the following line to the path of the configuration file
    src/train.py /your/path/LLaMA-Factory/examples/minicpm/minicpm_dpo.yaml
```

# Start Training

Finally, execute the training script from the `LLaMA-Factory` directory:
```sh
cd LLaMA-Factory
bash /your/path/LLaMA-Factory/examples/minicpm/single_node.sh
```