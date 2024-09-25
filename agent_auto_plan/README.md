# finetune minicpmv with some pairs having image and some not

1. git the code
```bash
git clone https://github.com/LDLINGLINGLING/MiniCPM_Series_Tutorial
```
2. repalce the modeling_minicpmv.py file in minicpmv2.6 with MiniCPM_Series_Tutorial/auto_plan/finetune_language.py/replace_file/modeling_minicpmv.py

3. prepare the data with having image(only one) and some not

4. finetune the model follow the (tutorial)[https://github.com/OpenBMB/MiniCPM-V/blob/main/finetune/readme.md]