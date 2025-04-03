import hailo_sdk_client as hailo
import os

# ONNX 파일 경로
onnx_path = "./yolov8n_epoch25_data4/best.onnx"

# 캘리브레이션 데이터셋 경로
calib_path = "./yolov8n_epoch25_data4/train/images"

# Hailo 모델 컴파일러 설정
compiler = hailo.HailoCompiler()

# ONNX 모델 파싱
parsed_model = compiler.parse_model(onnx_path)

# 캘리브레이션 데이터셋 설정
calib_set = hailo.CalibrationDataset(calib_path)

# 모델 최적화
optimized_model = compiler.optimize(parsed_model, calib_set)

# HEF 파일로 컴파일
hef_path = os.path.splitext(onnx_path)[0] + ".hef"
compiler.compile(optimized_model, target_platform="hailo8", output_path=hef_path)

print(f"HEF 파일 저장 경로: {hef_path}")