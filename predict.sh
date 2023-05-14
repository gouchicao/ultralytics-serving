if [ -z $1 ] || [ -z $2 ]; then
    echo "Usage: "
    echo "  predict.sh APP_NAME MODEL_NAME [SHOW_LABELS]"
    echo "Examples: "
    echo "  predict.sh platen-switch platen-switch-yolov8n.onnx"
    echo "  predict.sh handwritten-digit-recognition digit-yolov8n.pt"
    echo "  predict.sh test-paper-analysis test-paper-analysis-yolov8n.onnx"
    echo ""
    exit 1
fi

APP_NAME=$1
MODEL_NAME=$2
SHOW_LABELS=${3:-true}

echo "🚗 YOLO Predict APP_NAME:$APP_NAME MODEL_NAME:$MODEL_NAME SHOW_LABELS:$SHOW_LABELS"

yolo detect predict \
    project=private/$APP_NAME name=images/target \
    model=private/$APP_NAME/models/$MODEL_NAME \
    source=private/$APP_NAME/images/source \
    save=true exist_ok=true show_labels=$SHOW_LABELS


