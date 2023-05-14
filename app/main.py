from fastapi import FastAPI, Request
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from .routers import detect
from .config import Config
from .models import Models
from .utils import file


config = Config()

app = FastAPI(title=config.title, version=config.version, 
              docs_url=None, redoc_url=None)

app.include_router(detect.router, prefix="/detect", tags=["目标检测"])

# 通过配置动态增加 Router
# eval('app.include_router(detect.router, prefix="/detect", tags=["目标检测"])')
for statement in config.statements:
    print(f'Exec Router statement: {statement}')
    exec(statement)

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


@app.get("/")
async def index(request: Request):
    images = file.get_images('static/images/source')
    templates = Jinja2Templates(directory="static")
    return templates.TemplateResponse("index.html", 
                                      {"request": request, "title": app.title, "version": app.version, "images": images})


@app.on_event('startup')
def load_model():
    Models()
