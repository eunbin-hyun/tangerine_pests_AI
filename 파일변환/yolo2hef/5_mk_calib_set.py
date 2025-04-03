import os
import numpy as np
import cv2
from tqdm import tqdm  # tqdm 추가

def preprocess_image(image, target_size=(640, 640)):
    """이미지 전처리: 크기 조정 및 정규화"""
    # 이미지 크기 조정
    resized_image = cv2.resize(image, target_size)
    # 정규화 (0-255 범위를 0-1로 변환)
    normalized_image = resized_image / 255.0
    return normalized_image

# 이미지가 있는 디렉토리 경로
image_dir = "./yolov8n_epoch25_data4/train/images"

# 캘리브레이션 데이터셋을 위한 이미지 목록
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

# 캘리브레이션 데이터셋 생성
calib_dataset = []

# tqdm으로 진행 상황 표시
print(f"Processing {len(image_files)} images for calibration dataset...")
for image_file in tqdm(image_files, desc="Processing Images"):
    try:
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        if image is not None:
            preprocessed_image = preprocess_image(image)
            calib_dataset.append(preprocessed_image)
        else:
            print(f"Warning: Could not read image: {image_path}")
    except Exception as e:
        print(f"Error processing image {image_file}: {e}")

# NumPy 배열로 변환
if calib_dataset:
    calib_dataset = np.array(calib_dataset)
    # .npy 파일로 저장
    output_file = 'calib_set.npy'
    np.save(output_file, calib_dataset)
    print(f"Calibration dataset saved to {output_file} with shape: {calib_dataset.shape}")
else:
    print("Error: No valid images processed for calibration dataset.")
