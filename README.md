# 🍊 Tangerine Pests AI Robot

> Raspberry Pi 5와 Hailo-8L NPU에서 감귤 병해충을 탐지하는 엣지 AI 방제 로봇

`YOLOv8` · `Raspberry Pi 5` · `Hailo-8L` · `ONNX` · `Hailo DFC` · `OpenCV`

## 프로젝트 소개

제주 감귤 농장에서 발생하는 병해충을 카메라로 탐지하고, 임베디드 환경에서 실시간 추론할 수 있도록 AI 모델과 로봇 하드웨어를 연결한 캡스톤 프로젝트입니다. AI-Hub 데이터와 자체 촬영 이미지를 정리해 YOLOv8 모델을 학습하고, 학습 모델을 Hailo-8L용 HEF로 변환해 Raspberry Pi 5에서 실행하는 파이프라인을 구성했습니다.

| 구분 | 내용 |
|---|---|
| 목표 | 감귤 병해충 탐지 모델을 저전력 엣지 장치에서 실행 |
| 담당 | 데이터셋 구축·라벨링, YOLOv8 학습, 모델 변환·NPU 배포, 하드웨어 통합 |
| 하드웨어 | Raspberry Pi 5, Hailo-8L NPU, 카메라 |
| 결과 | 실제 감귤 병해 이미지에서 병해 종류와 confidence를 실시간 표시 |

## 시스템 구성

| 3D 모델링 | 처리 과정 |
|---|---|
| ![작품 3D 모델](https://github.com/user-attachments/assets/561bceac-5aa4-4ca8-92ed-3836c5d8cd89) | ![AI 처리 과정](https://github.com/user-attachments/assets/4103ab83-3d87-4683-a295-9508ccc65937) |
| 카메라·Raspberry Pi 5·Hailo-8L을 탑재한 로봇 설계 | 촬영 → 전처리 → YOLO 추론 → 결과 시각화 |

## AI 개발 과정

### 1. 데이터와 모델 학습

- AI-Hub 공개 데이터와 자체 촬영 이미지를 병합
- 감귤 병해충 클래스 라벨링 및 학습 데이터셋 구성
- YOLOv8 기반 탐지 모델 학습과 실제 농작물 이미지 검증
- Raspberry Pi 추론 환경을 고려해 입력 크기와 모델 규모 조정

### 2. 엣지 AI 양자화와 HEF 생성

GPU에서 학습한 PyTorch 모델은 Hailo-8L에서 바로 실행할 수 없습니다. 아래 단계를 거쳐 연산 그래프를 변환하고, Hailo Dataflow Compiler에서 후학습 양자화(PTQ)와 하드웨어 최적화를 수행한 뒤 NPU 실행 파일인 HEF를 생성했습니다.

```text
PyTorch checkpoint (.pt)
  → ONNX graph (.onnx)
  → parsed Hailo archive (.har)
  → PTQ / hardware optimization (quantized .har)
  → Hailo executable format (.hef)
```

| 단계 | 작업 | 저장소 파일 |
|---|---|---|
| 1 | YOLO 모델 구조와 weight를 ONNX로 export | [`3_pt2onnx.py`](./파일변환/yolo2hef/3_pt2onnx.py) |
| 2 | 입력·출력 노드를 확인해 변환 범위 검증 | [`4_checknode.py`](./파일변환/yolo2hef/4_checknode.py) |
| 3 | 학습 이미지를 640×640, 0–1 범위로 전처리해 calibration 배열 생성 | [`5_mk_calib_set.py`](./파일변환/yolo2hef/5_mk_calib_set.py) |
| 4 | ONNX 그래프를 Hailo-8L용 HAR로 parse | [`6_onnx2har.sh`](./파일변환/yolo2hef/6_onnx2har.sh) |
| 5 | calibration 분포를 기준으로 PTQ·레이어 최적화 | [`7_har2opt.sh`](./파일변환/yolo2hef/7_har2opt.sh) |
| 6 | 최적화된 HAR를 Hailo-8L 실행 파일 HEF로 compile | [`8_opt2hef.sh`](./파일변환/yolo2hef/8_opt2hef.sh) |

현재 `7_har2opt.sh`는 변환 경로 검증을 위해 `--use-random-calib-set` 옵션을 사용합니다. 실제 배포 정확도를 재현하려면 `5_mk_calib_set.py`로 만든 대표 데이터셋을 optimization 단계에 연결하고, 양자화 전·후 성능을 같은 테스트셋으로 비교해야 합니다.

### 3. Raspberry Pi 5 배포

- Raspberry Pi 5가 카메라 입력과 애플리케이션 흐름을 담당
- Hailo-8L이 HEF 모델의 추론을 가속
- 추론 결과를 병해 종류와 confidence로 시각화
- CPU 단독 추론의 부하를 줄이고 임베디드 실시간 처리 구조 구성

## 탐지 결과

<p align="center">
  <img src="https://github.com/user-attachments/assets/2fde7039-123d-4205-8278-411afc4c127a" width="680" alt="감귤 궤양병 실시간 탐지 결과" />
</p>

실제 병해충 이미지에서 `tangerines_canker: 0.90`과 같이 클래스와 confidence를 표시했습니다.

## 저장소 구조

```text
tangerine_pests_AI/
├─ AI학습/                 # 데이터 전처리, YOLO 학습·검증 노트북
├─ 파일변환/yolo2hef/      # PT → ONNX → HAR → HEF 변환 도구
└─ docs/                   # 특허 출원 자료
```

## 성과

- 2024 IP 창의발명 경진대회 **동상**
- 2024 제32회 설계 및 팀프로젝트 작품전시회 **최우수상**
- 2024-2학기 캡스톤디자인 결과발표회 **장려상**
- 특허 출원 — **감귤 병충해 실시간 진단 및 예방 장치** (`10-2025-0016861`)
  - [특허 출원서](./docs/10-2025-0016861_특허출원서.pdf)
