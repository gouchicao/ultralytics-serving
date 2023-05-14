import os
import yaml

class Config:
    class Model:
        def __init__(self, model):
            self.name = model.get("name", None)
            self.path = model.get("path", None)
            if self.path:
                self.path = os.path.join("asserts/models", self.path)
            self.task = model.get("task", "detect")
            self.imgsz = model.get("imgsz", 640)
            self.conf = model.get("conf", 0.5)
            self.iou = model.get("iou", 0.4)

    def __init__(self, config_file="app/config.yaml"):
        config = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)

        self.title = config.get("title", "Ultralytics Inference Serving API")
        self.version = config.get("version", "1.0.0")
        self.statements = [router["statement"] for router in config.get("routers", [])]
        self.models = [self.Model(model) for model in config.get("models", [])]

    def get_model_from_name(self, name):
        for model in self.models:
            if model.name == name:
                return model
        return None
