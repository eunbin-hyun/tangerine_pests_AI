#!/bin/bash

ONNX_FILE="./yolov8n_epoch25_data4/best.onnx"
HAR_FILE="./yolov8n_epoch25_data4/yolov8n.har"

hailo parser onnx $ONNX_FILE \
    --net-name yolov8n \
    --har-path $HAR_FILE \
    --start-node-names images \
    --end-node-names output0 \
    --hw-arch hailo8l
