# Ultralytics Serving
Inference service based on Ultralytics

## 创建虚拟环境
python -m venv venv
source venv/bin/activate

## 安装依赖包
### 创建 requirements.txt
```txt
fastapi
python-multipart
aiofiles
onnxruntime
ultralytics
uvicorn[standard]
gunicorn
pytest
httpx
```

### 安装
pip install -r requirements.txt

## 调试
创建 `launch.json` 文件，用于调试 FastAPI 应用。

.vscode/launch.json
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: FastAPI",
            "type": "python",
            "request": "launch",
            "module": "uvicorn",
            "args": [
                "app.main:app",
                "--reload"
            ],
            "jinja": true,
            "justMyCode": true
        }
    ]
}
```
* --reload 修改后可以自动加载，适用于开发。
* [Debug FastAPI in VS Code IDE](https://tutlinks.com/debug-fastapi-in-vs-code-ide/)
* [Debug FastAPI application in VSCode](https://stackoverflow.com/questions/60205056/debug-fastapi-application-in-vscode)

## 创建项目的目录结构
```
ultralytics-serving
├── app
│   ├── __init__.py
│   └── main.py
├── asserts/
├── models/
├── requirements.txt
├── routers
│   └── __init__.py
├── static/
├── tests
│   └── __init__.py
└── utils
    └── __init__.py
```

## 构建镜像
### Ultralytics Serving 镜像
#### amd64
```shell
docker buildx build --platform=linux/amd64 --pull --rm -f Dockerfile -t wangjunjian/ultralytics-serving:amd64 . --push
```

#### arm64
```shell
docker buildx build --platform=linux/arm64 --pull --rm -f Dockerfile -t wangjunjian/ultralytics-serving:arm64 . --push
```
* --pull 始终尝试拉取最新版本的镜像
* --rm 成功构建后删除中间容器
* --push 构建完成后推送到 Docker Hub

* [docker build](https://docs.docker.com/engine/reference/commandline/build/)
* [docker buildx build](https://docs.docker.com/engine/reference/commandline/buildx_build/)
* [如何构建多架构 Docker 镜像？](https://www.infoq.cn/article/V9Qj0fJj6HsGYQ0LpHxg)
* [Building Multi-Arch Images for Arm and x86 with Docker Desktop](https://www.docker.com/blog/multi-arch-images/)
* [在 M1 Mac 上构建 x86 Docker 镜像](https://prinsss.github.io/build-x86-docker-images-on-an-m1-macs/)

## 镜像测试
```shell
docker run --rm -it -p 80:80/tcp ultralytics-serving:amd64
docker run --rm -it -p 80:80/tcp ultralytics-serving:arm64
```

### curl
```shell
curl -X 'POST' \
  'http://localhost:8000/detect/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@asserts/images/catdog.jpg;type=image/jpeg'
```
* [How to install JQ on Mac on the command line?](https://stackoverflow.com/questions/37668134/how-to-install-jq-on-mac-on-the-command-line)

## 开发
### 环境
#### conda 降低 Python 版本，依赖包需要重新安装
```shell
conda install python=3.10.9
```

### 调试
#### macOS 上 F11 快捷键不能 Step Into
打开【系统设置】->【桌面与程序坞】->【快捷键】->【显示桌面】，设置为 `-`。
* [Unable to debug with F11 with VSCODE (toggled Fullscreen)?](https://stackoverflow.com/questions/29955146/unable-to-debug-with-f11-with-vscode-toggled-fullscreen)

### 模型
#### 模型可视化
查看模型的网络，层的名字，输入和输出的形状。
* [Netron](https://netron.app)

#### ONNXRuntime 推理 YOLO
```py
import cv2
import numpy as np

img = cv2.imread("image.jpg", cv2.IMREAD_UNCHANGED)
resized = cv2.resize(img, (640,640), interpolation = cv2.INTER_AREA).astype(np.float32)
resized = resized.transpose((2, 0, 1))
resized = np.expand_dims(resized, axis=0)  # Add batch dimension

import onnxruntime as ort
ort_session = ort.InferenceSession("yolov5.onnx", providers=["CUDAExecutionProvider"])
# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: resized}
ort_outs = ort_session.run(None, ort_inputs)
```
* [How to do inference with YOLOv5 and ONNX](https://stackoverflow.com/questions/75799955/how-to-do-inference-with-yolov5-and-onnx)
* [Yolov5 inferencing on ONNXRuntime and OpenCV DNN](https://medium.com/mlearning-ai/yolov5-inferencing-on-onnxruntime-and-opencv-dnn-d20e4c52dc31)
* [BlueMirrors/Yolov5-ONNX](https://github.com/BlueMirrors/Yolov5-ONNX/blob/main/detect.py)

#### ONNX 模型的输入与输出的名字
```py
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
```
* [ONNX Runtime Inference | session.run() multiprocessing](https://stackoverflow.com/questions/70803924/onnx-runtime-inference-session-run-multiprocessing)

### Python
#### eval和exec的区别
`eval()`和`exec()`都是Python内置函数，但它们的作用不同。

`eval()`函数用于计算表达式的值，返回表达式计算的结果。例如：

```python
x = 5
y = 3
result = eval('x + y')
print(result)
```

这个例子中，`eval()`函数计算字符串`'x + y'`的值，并将结果赋给变量`result`，最后输出`8`。

`exec()`函数用于执行一段Python代码，它没有返回值。例如：

```python
x = 5
y = 3
code = """
result = x + y
print(result)
"""
exec(code)
```

这个例子中，我们定义了一个字符串`code`，其中包含两个语句。然后我们用`exec()`函数执行了这个字符串，这样就会执行其中的两个语句，并输出结果`8`。

简而言之，`eval()`计算表达式的值并返回结果，而`exec()`执行一段Python代码，没有返回值。

#### 持久化保存临时文件
```py
file = tempfile.NamedTemporaryFile(delete=False)
```

#### 设置临时文件后缀
```py
file = tempfile.NamedTemporaryFile(suffix='.txt')
```
* [善用tempfile库创建python进程中的临时文件](https://www.cnblogs.com/dechinphy/p/tempfile.html)

#### 获得文件扩展名
```py
import pathlib

print(pathlib.Path('yourPath.example').suffix) # '.example'
print(pathlib.Path("hello/foo.bar.tar.gz").suffixes) # ['.bar', '.tar', '.gz']
```
* [Extracting extension from filename in Python](https://stackoverflow.com/questions/541390/extracting-extension-from-filename-in-python)

#### 其它格式的图像保存为指定的格式
```py
import cv2

# Load .png image
image = cv2.imread('image.png')

# Save .jpg image
cv2.imwrite('image.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
```
* [How to convert PNG to JPG in Python?](https://stackoverflow.com/questions/60048149/how-to-convert-png-to-jpg-in-python)

#### 数据的维度进行压缩，去掉维数为1的的维度
```py
#tensor([[[-0.1313, -1.0998, -1.9624]]])     torch.Size([1, 1, 3]) 
a=torch.randn(1,1,3)

#tensor([-0.1313, -1.0998, -1.9624])     torch.Size([3])
b=torch.squeeze(a)
```
* [pytorch学习 中 torch.squeeze() 和torch.unsqueeze()的用法](https://zhuanlan.zhihu.com/p/114653512)

#### 时间测量
```py
tic = time.perf_counter()
predict(img)
toc = time.perf_counter()
print(f"Predict {toc - tic:0.4f} seconds")
```
* [Python Timer Functions: Three Ways to Monitor Your Code](https://realpython.com/python-timer/)

#### 加载 yaml 文件
```py
config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
```
* [Python YAML: How to Load, Read, and Write YAML](https://python.land/data-processing/python-yaml)

#### gunicorn
* [loading models in FastAPI projects at startup](https://stackoverflow.com/questions/65636962/loading-models-in-fastapi-projects-at-startup)

#### 使用 OpenCV 移除图像中的黑色线段，仅保留红色数字
```python
import cv2

# 加载图像
img = cv2.imread('image.jpg')

# 将图像转换为HSV颜色空间
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 定义红色数字的HSV范围
lower_red = (0, 100, 100)
upper_red = (10, 255, 255)

# 创建掩码，仅保留红色数字
mask = cv2.inRange(hsv, lower_red, upper_red)

# 移除黑色线段
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
dilated_mask = cv2.dilate(mask, kernel, iterations=2)
eroded_mask = cv2.erode(dilated_mask, kernel, iterations=2)

# 应用掩码
result = cv2.bitwise_and(img, img, mask=eroded_mask)

# 显示结果
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

在这个例子中，我们首先加载图像，然后将其转换为HSV颜色空间。然后，我们定义红色数字的HSV范围，并创建一个掩码，仅保留红色数字。接下来，我们使用形态学操作（膨胀和腐蚀）移除黑色线段。最后，我们将掩码应用于原始图像，并显示结果。

### FastAPI
#### 在Swagger UI中隐藏接口
要在Swagger UI中隐藏某个接口，可以在FastAPI中使用装饰器`@app.delete`，`@app.put`，`@app.post`，`@app.get`，`@app.patch`等来定义路由。例如：

```python
@app.get("/items/{item_id}", tags=["items"], summary="Get an item")
async def read_item(item_id: int):
    return {"item_id": item_id, "name": "Fake Item"}

@app.post("/items/", tags=["items"], summary="Create an item", deprecated=True)
async def create_item(item: Item):
    return item
```

在上面的示例中，`@app.post`装饰器中使用了`deprecated=True`参数，表示这个接口已经过时。在Swagger UI中，这个接口将被标记为过时，并显示警告。如果想完全隐藏这个接口，可以在装饰器中添加`include_in_schema=False`参数。例如：

```python
@app.post("/items/", tags=["items"], summary="Create an item", include_in_schema=False)
async def create_item(item: Item):
    return item
```

这样，在Swagger UI中就不会显示这个接口。

#### OpenCV 直接读取 UploadFile
```py
@app.post("/analyze", response_model=Analyzer)
async def analyze_route(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
```
* [Receiving an image with Fast API, processing it with cv2 then returning it]()https://stackoverflow.com/questions/61333907/receiving-an-image-with-fast-api-processing-it-with-cv2-then-returning-it

#### 返回图像文件，使用内存中的图像
```py
import io
from starlette.responses import StreamingResponse

app = FastAPI()

@app.post("/vector_image")
def image_endpoint(*, vector):
    # Returns a cv2 image array from the document vector
    cv2img = my_function(vector)
    res, im_png = cv2.imencode(".png", cv2img)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")
```
* [How do I return an image in fastAPI?](https://stackoverflow.com/questions/55873174/how-do-i-return-an-image-in-fastapi)

#### 主页显示静态文件
```py
from fastapi import  FastAPI
from fastapi.staticfiles import StaticFiles

# Use this to serve a static/index.html
from starlette.responses import FileResponse 

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')
```
* [How can I serve static files (html, js) easily? #130](https://github.com/tiangolo/fastapi/issues/130)
* [A minimal fastapi example loading index.html](https://stackoverflow.com/questions/65916537/a-minimal-fastapi-example-loading-index-html)
* [Insert local image in the FastAPI automatic documentation](https://stackoverflow.com/questions/70740267/insert-local-image-in-the-fastapi-automatic-documentation)

#### 主页显示使用模板
```html
<html>
   <body>
      <h2>{{title}}</h2>
   </body>
</html>
```

```py
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


title = "Ultralytics Inference Serving API"

app = FastAPI(title=title, version="1.0.0")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "title": title})
```
* [FastAPI Templates](https://fastapi.tiangolo.com/advanced/templates/)
* [FastAPI - Quick Guide](https://www.tutorialspoint.com/fastapi/fastapi_quick_guide.htm)

#### 隐藏 /docs（Swagger UI） 页面上的 `Schemas`
`swagger_ui_parameters={"defaultModelsExpandDepth": -1}`
```py
app = FastAPI(title="Ultralytics Inference Serving API", version="1.0.0",
              swagger_ui_parameters={"defaultModelsExpandDepth": -1})
```
* [How to hide Schemas section in /docs? #2633](https://github.com/tiangolo/fastapi/issues/2633)

#### 修改图标（favicon）
* [cleanpng](https://www.cleanpng.com)
* [Extending OpenAPI](https://fastapi.tiangolo.com/advanced/extending-openapi/)
* [How to change FastAPI's swagger favicon?](https://dev.to/kludex/how-to-change-fastapis-swagger-favicon-4j6)
* [favicon with FastAPI?](https://stackoverflow.com/questions/68786428/setting-favicon-with-fastapi)

#### OpenAPI
* [Swagger UI](https://github.com/swagger-api/swagger-ui)
* [Redoc](https://github.com/Redocly/redoc)
* [Extending OpenAPI](https://fastapi.tiangolo.com/zh/advanced/extending-openapi/)
* [Swagger](https://swagger.io)

#### 在文档中增加代码示例
* [Is there support for x-codeSamples #8946](https://github.com/tiangolo/fastapi/discussions/8946)
* [How to Add Code Samples to ReDoc in FastAPI](https://levelup.gitconnected.com/how-to-add-code-samples-to-redoc-in-fastapi-1e7af847c4ef)

### HTML & CSS
#### 在不能联接网络（内网）时，访问 / 会出现下面的错误。
```html
INFO:     127.0.0.1:63009 - "GET /apple-touch-icon-precomposed.png HTTP/1.1" 404 Not Found
INFO:     127.0.0.1:63010 - "GET /apple-touch-icon.png HTTP/1.1" 404 Not Found
INFO:     127.0.0.1:63011 - "GET /favicon.ico HTTP/1.1" 404 Not Found
```

在 `<head>` 标签内加入下面的代码
```html
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black">
```
* [Keep getting 404's for apple-touch-icon.png](https://stackoverflow.com/questions/14986650/keep-getting-404s-for-apple-touch-icon-png)
* [FastAPI with uvicorn getting 404 Not Found error](https://stackoverflow.com/questions/65599686/fastapi-with-uvicorn-getting-404-not-found-error)

#### 在不能联接网络（内网）时，访问 /docs 或 /redoc 时会出现错误，且显示特别慢（分钟）。
```
[Error] Failed to load resource: 似乎已断开与互联网的连接。 (swagger-ui.css, line 0)
[Error] ReferenceError: Can't find variable: SwaggerUIBundle
[Error] Failed to load resource: 似乎已断开与互联网的连接。 (swagger-ui-bundle.js, line 0)
[Error] Failed to load resource: 似乎已断开与互联网的连接。 (favicon.png, line 0)
```

解决办法，注意 FastAPI 对象的参数：docs_url=None, redoc_url=None
```py
from fastapi import FastAPI
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.staticfiles import StaticFiles

app = FastAPI(docs_url=None, redoc_url=None)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
    )

@app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
async def swagger_ui_redirect():
    return get_swagger_ui_oauth2_redirect_html()

@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=app.title + " - ReDoc",
        redoc_js_url="/static/redoc.standalone.js",
    )
```

#### 超链接移除下划线
`style="text-decoration: none"`
```html
<a href="docs" style="text-decoration: none">交互式开发文档（Swagger UI）</a>
```

#### 图片居中显示
```html
<div>
   <table width="100%" height="100%" align="center" valign="center">
   <tr><td>
      <img src="foo.jpg" alt="foo" />
   </td></tr>
   </table>
</div>
```
* [How to make an image center (vertically & horizontally) inside a bigger div ](https://stackoverflow.com/questions/388180/how-to-make-an-image-center-vertically-horizontally-inside-a-bigger-div)

### Docker & Dockerfile
* [Dockerfile reference](https://docs.docker.com/engine/reference/builder/)
* [docker build](https://docs.docker.com/engine/reference/commandline/build/)
* [Dockerfile reference - ARG variable definition](https://docs.docker.com/engine/reference/builder/#arg)
* [docker buildx build](https://docs.docker.com/engine/reference/commandline/buildx_build/)

#### .dockerignore
```
**/.DS_Store # 构建 Docker 镜像时忽略所有目录下的 .DS_Store 文件。
```

### Jupyter Notebook
#### 显示图片
```py
from IPython.display import Image
display(Image(filename='test.png'))
```
* [How can I display an image from a file in Jupyter Notebook?](https://stackoverflow.com/questions/11854847/how-can-i-display-an-image-from-a-file-in-jupyter-notebook)

## 参考资料
* [FastAPI](https://fastapi.tiangolo.com/zh/)
* [更大的应用 - 多个文件](https://fastapi.tiangolo.com/zh/tutorial/bigger-applications/)
* [Ultralytics](https://github.com/ultralytics/ultralytics)
* [Ultralytics Python Usage](https://docs.ultralytics.com/usage/python/)
* [ONNX Runtime](https://onnxruntime.ai)
* [人工智能服务 REST API 响应的 JSON 格式](https://wangjunjian.com/restapi/2023/04/04/json-format-of-ai-service-rest-api-response.html)
* [gouchicao/restapi](https://github.com/gouchicao/restapi/blob/main/fastapi/app/routers/files.py)
* [Using Git source control in VS Code](https://code.visualstudio.com/docs/sourcecontrol/overview)
* [fastapi/tests/test_application.py](https://github.com/tiangolo/fastapi/blob/master/tests/test_application.py)
* [Python decorator to measure execution time](https://dev.to/kcdchennai/python-decorator-to-measure-execution-time-54hk)
* [python](https://hub.docker.com/_/python)
* [nvidia/cuda](https://hub.docker.com/r/nvidia/cuda)
* [EmojiDB](https://emojidb.org)
* [YAML](https://yaml.org)
* [YAML - Array & collection](https://www.w3schools.io/file/yaml-arrays/)
* [numpy数组升维函数expand_dims()](https://www.cnblogs.com/easternE/p/16622557.html)
* [Removing noisy lines from image - opencv - python](https://dsp.stackexchange.com/questions/52089/removing-noisy-lines-from-image-opencv-python)
* [Having trouble understanding the code for this part. #7769](https://github.com/ultralytics/yolov5/issues/7769)
* [Introduction to OpenCV inRange](https://www.educba.com/opencv-inrange/)
* [How to detect two different colors using `cv2.inRange` in Python-OpenCV?](https://stackoverflow.com/questions/48109650/how-to-detect-two-different-colors-using-cv2-inrange-in-python-opencv)
* [YOLO Classify](https://docs.ultralytics.com/tasks/classify/?h=classify)
* [MNIST 手写数字识别](https://kittenbot.readthedocs.io/Tensorflow/MnistTraining.html)
