# 构建开发环境，安装依赖包
# [python](https://hub.docker.com/_/python)
FROM python:3.10 AS builder

ENV APP_HOME=/ultralytics-serving

WORKDIR ${APP_HOME}

# 提前安装，因为 cpu 版本需要指定 index-url。
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

COPY ./requirements.txt ${APP_HOME}/requirements.txt
RUN pip install --no-cache-dir --upgrade -r ${APP_HOME}/requirements.txt

# 编译应用
COPY ./app ${APP_HOME}/app
RUN find ${APP_HOME}/app -name '*.py[co]' -delete \
    && python -m compileall -b ${APP_HOME}/app \
    && find ${APP_HOME}/app -name '*.py' -delete


# 发布应用
FROM python:3.10

ENV APP_HOME=/ultralytics-serving

WORKDIR ${APP_HOME}

RUN apt-get update \
    && apt-get install -y libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder ${APP_HOME}/app ${APP_HOME}/app

EXPOSE 80

COPY ./asserts ${APP_HOME}/asserts
COPY ./static ${APP_HOME}/static

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
# CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:80"]
