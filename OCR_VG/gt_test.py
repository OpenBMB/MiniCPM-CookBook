from chat import MiniCPMVChat, img2base64
import torch
import json
import cv2
import os
torch.manual_seed(0)
import re
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def parse_text(input_str):
    # 使用正则表达式匹配文本和数字
    pattern = r'<ref>(.*?)<box>(\d+)</box><box>(\d+)</box><box>(\d+)</box><box>(\d+)</box></ref>'
    matches = re.findall(pattern, input_str)
    
    result = []
    for match in matches:
        text = match[0]
        numbers = [int(num) for num in match[1:]]
        # 将每对数字组成坐标
        coordinates = [[numbers[i], numbers[i+1]] for i in range(0, len(numbers), 2)]
        result.append({text: coordinates})
    
    return result
def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "/root/ld/ld_project/MiniCPM-V/finetune/test/simsun.ttc", textSize
    )
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
def draw(final_box,final_text,img,height,width):
    color = (0, 0, 255)
    # 绘制每个矩形
    for r_index,rectangle in enumerate(final_box):
        # PIL需要左上角和右下角的坐标来绘制矩形
        left_top = (int(rectangle[0][0]*width/1000), int(rectangle[0][1]*height/1000))
        right_bottom = (int(rectangle[1][0]*width/1000), int(rectangle[1][1]*height/1000))

        # 使用draw.rectangle()函数绘制矩形
        cv2.rectangle(img,left_top, right_bottom, color, thickness=2)
        text = final_text[r_index]  # 要显示的文字
        text='\n'.join([text[i:i+40] for i in range(0, len(text), 40)])
        #text = "开头五字"+text[:5]+"结尾五字"+text[-5:] if len(text)>10 else text
        img = cv2ImgAddText(img, text, left_top[0], left_top[1], (75, 0 , 130), 30)

    # 在屏幕上显示带有矩形框的图片
    cv2.imwrite(os.path.join(out_path,pic_path.split('/')[-1]), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return
def get_pic_path(dir):
    if os.path.isdir(image_path):
        jpg_files = []
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith('.jpg'):
                    jpg_files.append(os.path.join(root, file))
    else:
        jpg_files = [image_path]
    return jpg_files
if __name__ == '__main__':
    image_path='/root/ld/ld_project/MIniCPM_Series_Tutorial/OCR_VG/input' # 测试图片路径
    out_path='/root/ld/ld_project/MIniCPM_Series_Tutorial/OCR_VG/out' #输出图片保存文件夹
    chat_model = MiniCPMVChat('/root/ld/ld_project/MiniCPM-V/finetune/output/merge_MiniCPM-Llama3-V-2_5') # 模型路径
    pic_path_list=get_pic_path(image_path)
    # with open('/root/ld/ld_dataset/2022_12_SCUT-HCCDoc_Val/vg_box_test.json', 'r') as file:
    #     data = json.load(file)
    #     pic_path_list=[i["image"] for i in data]    
    for pic_path in pic_path_list:
        im_64 = img2base64(pic_path)
        # First round chat 
        msgs = [{"role": "user", "content": "识别图中的文字,并且输出位置"}]
        inputs = {"image": im_64, "question": json.dumps(msgs)}
        answer = chat_model.chat(inputs)
        img = cv2.imread(pic_path)
        height, width = img.shape[:2]
        format_output=parse_text(answer)
        final_text,final_box = [list(i.keys())[0] for i in format_output ],[list(i.values())[0] for i in format_output]
        draw(final_box,final_text,img,height,width)
