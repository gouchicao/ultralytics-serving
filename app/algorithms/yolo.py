from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator, colors
from app.base_model import Box, DetectObject


class YOLOV8:
    def load(self, model_config) -> None:
        self.model_config = model_config
        self.model = YOLO(model=model_config.path, task=model_config.task)

    def classify(self, source):
        results = self.model(source=source, conf=self.model_config.conf)
        result = results[0]
        probs = result.probs
        top1 = probs.argsort(0, descending=True)[:1].tolist()
        label = result.names[top1[0]]
        prob = probs[top1[0]]
        return label, prob

    def predict(self, source) -> list[DetectObject]:
        objects = []

        results = self.model(source=source, conf=self.model_config.conf)
        for result in results:
            for d in reversed(result.boxes):
                c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())

                # xywh = d.xywh.squeeze() # xywh 计算有问题，不能使用。
                x1, y1, x2, y2 = d.xyxy.squeeze()
                box = Box(x=x1, y=y1, w=x2-x1, h=y2-y1)
                objects.append(DetectObject(score=conf, label=result.names[c], box=box))

        return objects
    
    def demo(self, source):
        objects = self.model(source=source, conf=self.model_config.conf)
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
