## Description: Build docker images for ultralytics-serving

build_image_and_push() {
    local dockerfile=$1
    local app_name=$2
    local platforms=$3

    for platform in $platforms
    do
        echo "🐳 Building $app_name:$platform"
        # 避免与 Docker Hub 通信，加快构建速度。
        # --pull 从 Docker Hub 拉取最新的基础镜像
        # --push 推送到 Docker Hub
        # --rm 成功构建后删除中间容器
        # docker buildx build --platform=linux/$platform --pull --rm -f $dockerfile -t wangjunjian/$app_name:$platform "." --push
        docker buildx build --progress=plain --platform=linux/$platform --rm -f $dockerfile -t wangjunjian/$app_name:$platform "."
        echo "💯\n"
    done
}

build_project_image() {
    local app_name=$1
    local platforms=$2
    local workdir=private/$app_name
    local dockerfile=private/Dockerfile

    for platform in $platforms
    do
        echo "🐳 Building $app_name:$platform"
        # docker buildx build --platform=linux/$platform --build-arg APP_NAME=$app_name --build-arg PLATFORM=$platform --pull --rm -f $dockerfile -t $app_name:$platform $workdir
        docker buildx build --platform=linux/$platform --build-arg APP_NAME=$app_name --build-arg PLATFORM=$platform --rm -f $dockerfile -t $app_name:$platform $workdir
        echo "💯\n"
    done
}


# 只有当用户直接执行当前脚本才会执行下面的镜像构建，避免了 source build_dockerfile.sh 也会执行镜像构建的问题
if [ $0 = "build_dockerfile.sh" ]; then
    ## Ultralytics Serving
    APP_NAME=ultralytics-serving
    PLATFORMS=(amd64 arm64)
    build_image_and_push Dockerfile $APP_NAME "${PLATFORMS[*]}"
fi