# MiniCPM-V Finetuning


We offer the demo scripts for easy finetuning of the pretrained **MiniCPM-Llama3-V 2.5** on both language data and multimodal data. Our finetune scripts use transformers Trainer and DeepSpeed by default.

### Replace your modeling file
You should use modeling_minicpmv.py and resample.py under /MiniCPM-V/finetune/replace_file/ to replace the files with the same name in your MiniCPM-Llama3-V-2_5 download path

### Data preparation

To prepare your finetuning data, you should formulate each sample as a dictionary consisting of an id, an image path list with an image, and a list of conversations. Then save data samples in JSON files.

For the vision-language example with image, you are required to provide **\<image\>** to define the position to insert the image embeddings. 

<details>
  <summary>
    <b>vision-language example (vl_finetune_data.json) with 1 samples.</b>
  </summary>

```
  [
    {
      "id": "0",
      "image": 'path/to/image_0.jpg',
      "conversations": [
            {
              'role': 'user', 
              'content': '<image>\nHow many desserts are on the white plate?'
            }, 
            {
                'role': 'assistant', 
                'content': 'There are three desserts on the white plate.'
            },   
            {
                'role': 'user', 
                'content': 'What type of desserts are they?'
            },
            {
                'role': 'assistant', 
                'content': 'The desserts are cakes with bananas and pecans on top. They share similarities with donuts, but the presence of bananas and pecans differentiates them.'
            }, 
            {
                'role': 'user', 
                'content': 'What is the setting of the image?'}, 
            {
                'role': 'assistant', 
                'content': 'The image is set on a table top with a plate containing the three desserts.'
            },
        ]
    },
  ]
```

For the only-language example with image, you should not provide the <image>,
Everything else is the same.

<details>
  <summary>
    <b>only-language example (vl_finetune_data.json) with 1 samples.</b>
  </summary>

```
  [
    {
      "id": "0",
      "conversations": [
            {
                "role": "user",
                "content": "我当前33岁，面部已出现大量雀斑，我推测这可能是遗传因素所致。我对此表示忧虑，雀斑的形成确有遗传倾向吗？有何治疗方法？"
            },
            {
                "role": "assistant",
                "content": "通常，雀斑的产生与体内内分泌失调及新陈代谢状态不佳有关，而非单纯的遗传因素。您与您的母亲在生活环境中以及日常习惯上的相似之处，可能共同促成了雀斑的生成。建议您注重规律作息，避免过度疲劳，减轻心理压力，并定期敷用面膜，以助于减轻雀斑的显现。"
            },
            {
                "role": "user",
                "content": "应当预约哪个科室进行诊疗？涉及的病症名称是什么？"
            },
            {
                "role": "assistant",
                "content": "应预约皮肤性病科进行就诊，相关病症可能为雀斑。"
            }
        ]
    },
  ]
```

</details>

### Full-parameter finetuning

Full-parameter parameter finetuning requires updating all parameters of LLM in the whole training process. Please specify the correct MODEL path, DATA path and LLM_TYPE in the shell scripts.

```shell
MODEL="openbmb/MiniCPM-Llama3-V-2_5" # or openbmb/MiniCPM-V-2
DATA="path/to/trainging_data" # json file
EVAL_DATA="path/to/test_data" # json file
LLM_TYPE="llama3" # if use openbmb/MiniCPM-V-2, please set LLM_TYPE=minicpm
```

To launch your training, run the following script:

```
sh finetune_ds.sh
```

Specially, Llama3 has a different chat_template for training and inference, we modified the chat_template for training, so please take care to restore the chat_template when inference on the training ckpt.

### LoRA finetuning

The LoRA allows light-weight model tuning with only a small subset of parameters updated. We provide the LoRA implementation based on `peft`. To launch your training, run the following script:

```
sh finetune_lora.sh
```

After training, you could load the model with the path to the adapter. We advise you to use absolute path for your pretrained model. This is because LoRA only saves the adapter and the absolute path in the adapter configuration json file is used for finding out the pretrained model to load.

```
from peft import AutoPeftModelForCausalLM

path_to_adapter="path_to_adapter"

model = AutoPeftModelForCausalLM.from_pretrained(
    # path to the output directory
    path_to_adapter,
    device_map="auto",
    trust_remote_code=True
).eval()

vpm_resampler_embedtokens_weight = torch.load(f"{path_to_adapter}/vpm_resampler_embedtokens.pt")

msg = model.load_state_dict(vpm_resampler_embedtokens_weight, strict=False)
```


