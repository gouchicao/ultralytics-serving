import threading
from .config import Config
from .algorithms.yolo import YOLOV8


class Models:
    _instance = None
    _lock = threading.Lock()
    _models = None

    def __new__(cls):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    # 加载模型
                    models = []
                    config = Config()
                    for i, model_config in enumerate(config.models):
                        print(f'Loading model name: {model_config.name} path: {model_config.path}')

                        model = YOLOV8()
                        model.load(model_config)
                        models.append(model)
                    cls._models = models
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    @staticmethod
    def getInstance():
        return Models()._instance
    
    def __getitem__(self, index):
        return self._models[index]
