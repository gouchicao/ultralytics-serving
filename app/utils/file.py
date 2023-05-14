import os
import cv2
import numpy as np
from fastapi import UploadFile

async def read_image_from_upload_file(file: UploadFile):
    contents = await file.read()
    nparray = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparray, cv2.IMREAD_COLOR)
    return img

def convert_to_gray_with_filter_color(img, lower_color, upper_color):
    # 转换颜色空间为HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 设定颜色范围
    img_mask = cv2.inRange(hsv, lower_color, upper_color)        
    
    # 灰度图像转换为三通道
    img_mask = cv2.cvtColor(img_mask.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    return img_mask

def get_images(path):
    image_paths = []

    ext_names = ['.png', '.jpg', '.jpeg']

    for filename in os.listdir(path):
        _, ext_name = os.path.splitext(filename.lower())
        if ext_name.lower() in ext_names:
            image_paths.append(filename)

    return image_paths
