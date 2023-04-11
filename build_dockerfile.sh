## Description: Build docker images for ultralytics-serving

build_image_push() {
    local dockerfile=$1
    local app_name=$2
    local platforms=$3

    for platform in $platforms
    do
        echo "🐳 Building $app_name:$platform"
        docker buildx build --platform=linux/$platform --pull --rm -f $dockerfile -t wangjunjian/$app_name:$platform "." --push
        echo "💯\n"
    done
}

build_local_image() {
    local dockerfile=$1
    local app_name=$2
    local platforms=$3
    local workdir=$4

    for platform in $platforms
    do
        echo "🐳 Building $app_name:$platform"
        docker buildx build --platform=linux/$platform --pull --rm -f $dockerfile.$platform -t $app_name:$platform $workdir
        echo "💯\n"
    done
}


PLATFORMS=(amd64 arm64)

## Ultralytics Serving
APP_NAME=ultralytics-serving
build_image_push Dockerfile $APP_NAME "${PLATFORMS[*]}"
