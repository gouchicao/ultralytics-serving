import collections
import io
import time
import cv2
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator, colors
from ..utils.file import read_image_from_upload_file
from app import main


router = APIRouter()


class Box(BaseModel):
    x: int
    y: int
    w: int
    h: int

    def center(self):
        return ( self.x+self.w//2, self.y+self.h//2 )
        
    def is_same_row(self, box):
        center_y = self.center()[1]
        return (center_y > box.y) and (center_y < box.y+box.h)

    def is_same_col(self, box):
        center_x = self.center()[0]
        return (center_x > box.x) and (center_x < box.x+box.w)

class DetectObject(BaseModel):
    score: float
    label: str
    box: Box

class ImageMetadata(BaseModel):
    width: int
    height: int
    type: str
    size: int

class PredictResult(BaseModel):
    objects: list[DetectObject]
    metadata: ImageMetadata
    time_ms: int

class DetectObjectWithRowColumn(DetectObject):
    row: int
    column: int


class Model:
    def load(self, config) -> None:
        self.config = config
        self.model = YOLO(model=config['model'], task=config['task'])

    def predict(self, source) -> list[DetectObject]:
        objects = []

        results = self.model(source=source, conf=self.config['conf'])
        for result in results:
            for d in reversed(result.boxes):
                c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())

                xywh = d.xywh.squeeze()
                box = Box(x=xywh[0], y=xywh[1], w=xywh[2], h=xywh[3])
                objects.append(DetectObject(score=conf, label=result.names[c], box=box))

        return objects
    
    def demo(self, source):
        objects = self.model(source=source, conf=self.config['conf'])
        for obj in objects:
            annotator = self._get_annotator(obj.orig_img, obj.names)
            for d in reversed(obj.boxes):
                c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
                name = ('' if id is None else f'id:{id} ') + obj.names[c]
                label = f'{name} {conf:.2f}'
                annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))
            return obj.orig_img
        
        return None
    
    def _get_annotator(self, img, cls_names):
        return Annotator(img, line_width=2, example=str(cls_names))

class RecognitionRowColumn:
    def recognition(self, objects: list[DetectObject]) -> list[DetectObjectWithRowColumn]:
        recognition_objects = []
        for obj in objects:
            recognition_objects.append(DetectObjectWithRowColumn(score=obj.score, label=obj.label, box=obj.box, row=0, column=0))

        # 相同行的对象归类
        # [[obj1, obj2], [obj3, obj4, obj5]]
        obj_rows = []
        for obj in recognition_objects:
            if not obj_rows:
                obj_rows.append([obj])
                continue

            is_found = False
            for obj_row in obj_rows:
                if obj.box.is_same_row(obj_row[0].box):
                    obj_row.append(obj)
                    is_found = True
                    break

            if not is_found:
                obj_rows.append([obj])

        # 行排序。使用y坐标进行排序
        y_objrows = {}
        for obj_row in obj_rows:
            y_objrows[obj_row[0].box.y] = obj_row

        y_objrows = collections.OrderedDict(sorted(y_objrows.items()))

        for r, obj_row in enumerate(y_objrows.values()):
            # 列排序。使用x坐标进行行内的对象排序
            x_objs = {}
            for obj in obj_row:
                x_objs[obj.box.x] = obj

            x_objs = collections.OrderedDict(sorted(x_objs.items()))
            for c, obj in enumerate(x_objs.values()):
                obj.row = r + 1
                obj.column = c + 1

        # 纠正错位的对象，使其与正确的列对应。
        target_row_idx, target_row = self._get_row_with_most_objects(y_objrows.values())
        for r, obj_row in enumerate(y_objrows.values()):
            if r == target_row_idx:
                continue

            for obj in obj_row:
                for target_obj in target_row:
                    if obj.box.is_same_col(target_obj.box):
                        obj.column = target_obj.column
                        break

        return recognition_objects

    def _get_detect_object_with_row_column(self, obj: DetectObject) -> DetectObjectWithRowColumn:
        return DetectObjectWithRowColumn(score=obj.score, label=obj.label, box=obj.box, row=0, column=0)

    def _get_row_with_most_objects(self, objrows):
        idx, row, objs_num = 0, None, 0

        for r, obj_row in enumerate(objrows):
            num = len(obj_row)
            if num > objs_num:
                idx, row, objs_num = r, obj_row, num
                
        return idx, row


model = Model()

@router.on_event('startup')
def load_model():
    model_name = main.config['model']
    print(f'Loading model: {model_name}')
    model.load(main.config)

@router.post("/predict")
async def predict(file: UploadFile = File(...)) -> PredictResult:
    """
    通过 multipart/form-data 上传图片文件，返回检测结果。
    """
    
    img = await read_image_from_upload_file(file)
    metadata = ImageMetadata(width=img.shape[1], height=img.shape[0], type=file.content_type, size=file.size)

    start_time = time.perf_counter()
    objects = model.predict(img)
    predict_time = int((time.perf_counter() - start_time) * 1000)

    return PredictResult(objects=objects, metadata=metadata, time_ms=predict_time)

@router.post('/demo', response_class=StreamingResponse)
async def demo(file: UploadFile = File(...)) -> StreamingResponse:
    """
    通过 multipart/form-data 上传图片文件，预测并进行标注，返回 JPEG 文件。
    """
    
    img = await read_image_from_upload_file(file)
    img = model.demo(img)
    res, img_jpg = cv2.imencode(".jpg", img)

    return StreamingResponse(io.BytesIO(img_jpg.tobytes()), media_type="image/jpeg")

@router.post('/recognition_row_column')
async def recognition_row_column(objects: list[DetectObject]) -> list[DetectObjectWithRowColumn]:
    recognition = RecognitionRowColumn()
    return recognition.recognition(objects)


if __name__ == "__main__":
    pass
