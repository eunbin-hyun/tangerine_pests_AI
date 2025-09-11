# tangerine_pests_AI
### 감귤병해충 판별 AI 로봇

![작품3D모델_정사각형2](https://github.com/user-attachments/assets/561bceac-5aa4-4ca8-92ed-3836c5d8cd89) | ![처리과정](https://github.com/user-attachments/assets/4103ab83-3d87-4683-a295-9508ccc65937)
---|---|
3D 모델링 | 처리과정

라즈베리파이5와 ai가속기(Hailo8L)을 이용한 감귤병충해 ai 판별 기기

yolov8n를 이용해 객체를 인식함

<br />

![howai](https://github.com/user-attachments/assets/9f780004-55ea-4822-b853-405b9c2199ed)



### 파일변환을 통해 AI와 HW결합

  AI학습파일(pt) -> onnx파일 -> hef파일 

<br />

### AI인식
![1](https://github.com/user-attachments/assets/de24dcf7-7528-4f33-93fb-3340a995757d)

# 🍊 tangerine_pests_AI
감귤병해충 방제 AI 로봇

## 프로젝트 개요
- **목적** : 제주 감귤농장에서 발생하는 병해충을 AI로 실시간 탐지·분류
- **주요 기술** : YOLOv8 Segmentation + Raspberry Pi + Hailo-8L NPU
- **역할** : 데이터셋 구축, AI 모델 학습 및 HW 통합

---

## 📐 System Architecture
| 3D 모델링 | 처리 과정 |
|-----------|-----------|
![작품3D모델_정사각형2](https://github.com/user-attachments/assets/561bceac-5aa4-4ca8-92ed-3836c5d8cd89) | ![처리과정](https://github.com/user-attachments/assets/4103ab83-3d87-4683-a295-9508ccc65937)
---|---|
| **로봇 하드웨어** : 카메라, Raspberry Pi 5, Hailo-8L NPU | **AI 파이프라인** : 촬영→데이터 전처리→YOLOv8 추론→결과 시각화 |

---

## 🧠 AI 개발 과정
| 데이터 구축 | AI 모델 학습 | AI 모델 적용 |
|-------------|-------------|-------------|
| 자체 촬영 + 라벨링 | YOLOv8 Segmentation 학습 | Hailo DFC/HEF 변환 및 임베디드 추론 |

---

## ⚙️ Hardware Integration
- Raspberry Pi 5 + Hailo-8L NPU
- pt → onnx 변환 → hef 파일 최적화
- 실시간 탐지 FPS: **xx.xx** (실험치)

---

## 🔎 실험 결과
| 감귤 궤양병 실측 | 탐지 결과 |
|-----------------|----------|
| ![원본 이미지](이미지경로) | ![AI 결과](이미지경로) |
| 실제 병해충 이미지를 실시간으로 탐지 및 확률 표시 (ex. `tangerines_canker:0.90`) |

---

## 🏆 Achievements & Publications
- **수상** : 2024 지식재산(IP) 창업·발명 경진대회 **동상**  
  - 감귤병해충 방제 AI 로봇 개발 및 상용화 제안
- **특허 출원** : **딥러닝 기반 병해충 감지 장치 및 방법**  
  - 출원번호 : `10-2024-XXXXX`  
  - [출원서 (PDF)](docs/tangerine_pests_patent.pdf)

---

## 💻 Tech Stack
| Hardware | Software |
|----------|----------|
| **Raspberry Pi 5** – 실시간 추론 및 제어 | **YOLOv8 Segmentation** – 병해충 탐지 |
| **Hailo-8L NPU** – AI 가속 및 모델 최적화 | **OpenCV** – 이미지 전처리 및 시각화 |
| **Camera Module** – 실시간 촬영 | **ONNX / Hailo DFC** – 모델 변환 및 HEF 빌드 |

