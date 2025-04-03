#!/bin/bash

hailo optimize yolov8n.har --hw-arch hailo8l --output-har-path yolov8n_quantized_model.har --use-random-calib-set