import collections
import io
import time
import cv2
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import StreamingResponse
from app.base_model import DetectObject, ImageMetadata, PredictResult, DetectObjectWithRowColumn
from app.algorithms.recognition_row_column import RecognitionRowColumn
from app.utils.file import read_image_from_upload_file
from app.models import Models


router = APIRouter()


@router.post("/predict", summary="预测")
async def predict(file: UploadFile = File(...)) -> PredictResult:
    """
    通过 multipart/form-data 上传图片文件，返回检测结果。
    """
    
    img = await read_image_from_upload_file(file)
    metadata = ImageMetadata(width=img.shape[1], height=img.shape[0], type=file.content_type, size=file.size)

    start_time = time.perf_counter()
    objects = Models()[0].predict(img)
    predict_time = int((time.perf_counter() - start_time) * 1000)

    return PredictResult(objects=objects, metadata=metadata, time_ms=predict_time)


@router.post('/demo', summary="演示", response_class=StreamingResponse)
async def demo(file: UploadFile = File(...)) -> StreamingResponse:
    """
    通过 multipart/form-data 上传图片文件，预测并进行标注，返回 JPEG 文件。
    """
    
    img = await read_image_from_upload_file(file)
    img = Models()[0].demo(img)
    res, img_jpg = cv2.imencode(".jpg", img)

    return StreamingResponse(io.BytesIO(img_jpg.tobytes()), media_type="image/jpeg")


@router.post('/recognition_row_column', summary="识别行号和列号")
async def recognition_row_column(objects: list[DetectObject]) -> list[DetectObjectWithRowColumn]:
    recognition = RecognitionRowColumn()
    return recognition.recognition(objects)
