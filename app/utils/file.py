import cv2
import numpy as np
from fastapi import UploadFile

async def read_image_from_upload_file(file: UploadFile):
    contents = await file.read()
    nparray = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparray, cv2.IMREAD_COLOR)
    return img
