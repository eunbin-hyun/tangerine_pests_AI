import torch
from ultralytics import YOLO

# 모델 로드
model = YOLO("./yolov8n_epoch25_data4/best.pt")

# 원하는 출력 노드 이름을 지정하며 ONNX 변환
onnx_path = model.export(
    format="onnx",
    opset=11
)

print(f"ONNX 모델 저장 경로: {onnx_path}")
