# FBM Defect Pattern Multi-Label Classification

Wafer chip **Fail Bit Map(FBM)** 이미지에서 불량 패턴을 감지하는 Multi-Label CNN 분류 모델입니다.

단일 패턴뿐 아니라 **2개 이상의 불량 패턴이 중첩된 이미지**에서도 각 패턴을 독립적으로 감지할 수 있습니다.

---

## Problem Statement

기존 Binary Classification 방식(패턴별 개별 모델 학습 → 각각 추론 후 결과 취합)은 **단일 패턴** 감지에서는 잘 동작하지만, **2개 패턴이 동시에 존재하는 중첩 불량**에서 모든 패턴을 정확히 감지하지 못하는 한계가 있었습니다.

이를 해결하기 위해:

1. **Multi-Label Classification** 아키텍처 도입 (Sigmoid + BCEWithLogitsLoss)
2. 단일 패턴 + **중첩 패턴 합성 데이터** 동시 학습
3. 중첩 패턴 전용 테스트셋으로 성능 검증

---

## Defect Pattern Classes (6종)

| Class | Description | 설명 |
|---|---|---|
| `row_line` | Horizontal line defects | 로우 방향 라인 불량 |
| `col_line` | Vertical line defects | 컬럼 방향 라인 불량 |
| `corner_rect` | Corner rectangular defects | 모서리 사각형 불량 |
| `nail` | Nail/crescent-shaped defects | 손톱/반달 형태 불량 |
| `edge` | Edge region defects | 가장자리 불량 |
| `block` | Block-shaped defects | 블록 형태 불량 |

- **Normal** = 모든 레이블이 0인 경우 (불량 패턴 미감지)
- **Composite** = 2개 이상 레이블이 1 (중첩 불량)

---

## Model Architecture

경량 CNN 기반 Multi-Label 분류기 (총 390,342 파라미터)

```
Input: 1 x 38 x 128 (Grayscale FBM image)
  │
  ├── Conv2d(1→32) + BN + ReLU + MaxPool
  ├── Conv2d(32→64) + BN + ReLU + MaxPool
  ├── Conv2d(64→128) + BN + ReLU + MaxPool
  ├── Conv2d(128→256) + BN + ReLU + AdaptiveAvgPool(1)
  │
  ├── Dropout(0.3)
  └── Linear(256→6)  ← raw logits

Output: 6 logits → Sigmoid → Threshold(0.5) → Multi-label prediction
```

- **Loss**: `BCEWithLogitsLoss` (각 클래스 독립 Binary Cross Entropy)
- **Optimizer**: Adam (lr=0.001)
- **GPU 최적화**: Mixed Precision (AMP/FP16), cuDNN benchmark

---

## Training Results

### Environment
- **GPU**: NVIDIA GeForce RTX 3080 (10GB VRAM)
- **CUDA**: 12.1 / PyTorch 2.5.1+cu121
- **Training Time**: 38.4 sec (30 epochs)

### Dataset
| Split | Images | Description |
|---|---|---|
| Train | 2,310 | 단일(1,440) + 중첩(630) + 정상(240) |
| Val | 555 | 단일 + 중첩 + 정상 |
| Test (Composite) | 135 | 중첩 패턴만 (C(6,2)=15 조합 × 9장) |

### Validation Performance
| Metric | Score |
|---|---|
| **Label Accuracy** (개별 레이블) | **99.1%** |
| **Exact Match** (6개 레이블 전체 일치) | **94.6%** |

### Composite Pattern Test (핵심 평가)

> 모든 이미지가 **2개 불량 패턴이 동시에 존재**하는 데이터 (135장)

| Metric | Score |
|---|---|
| **Exact Match** (2개 패턴 모두 정확 감지) | **80.0%** |
| **Label Accuracy** (개별 레이블 정확도) | **96.4%** |

### Per-Pattern Accuracy on Composite Test
| Pattern | Accuracy |
|---|---|
| block | 99.3% |
| col_line | 98.5% |
| edge | 98.5% |
| row_line | 97.0% |
| corner_rect | 95.6% |
| nail | 89.6% |

---

## Project Structure

```
fbm_classification/
├── README.md                 # 프로젝트 문서
├── requirements.txt          # 의존성 패키지
├── fbm_model.py              # CNN 모델 정의
├── generate_fbm_data.py      # 합성 FBM 데이터 생성 (단일 + 중첩)
├── train.py                  # Multi-Label 학습 + 평가
├── detect.py                 # CLI 추론 스크립트
└── webcam_detect.py          # GUI(tkinter) 추론 애플리케이션
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

> NVIDIA GPU 사용 시 CUDA 버전에 맞는 PyTorch 설치:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 2. Generate Synthetic Data
```bash
python generate_fbm_data.py
```
- 단일 패턴 (6종 × 300장) + 정상 (300장)
- 중첩 패턴 (15조합 × 60장 = 900장)
- 총 3,000장 → Train/Val/Test 분할
- 샘플 이미지: `data/fbm_dataset/samples/`

### 3. Train Model
```bash
python train.py --epochs 30
```
- GPU 자동 감지 (CUDA 사용 가능 시 GPU 학습)
- 학습 완료 후 중첩 테스트셋 자동 평가
- 모델 저장: `runs/fbm_train/best.pt`

### 4. Run Inference

**CLI 방식:**
```bash
# 단일 이미지
python detect.py --source path/to/fbm_image.png

# 폴더 전체
python detect.py --source data/fbm_dataset/test_composite/images/

# 임계값 조정
python detect.py --source path/to/image.png --threshold 0.3
```

**GUI 방식:**
```bash
python webcam_detect.py
```
- 파일 열기 / 폴더 일괄 분류
- 실시간 Threshold 슬라이더
- 패턴별 확률 시각화

---

## Key Design Decisions

### Multi-Label vs Multi-Class
| | Multi-Class (기존) | Multi-Label (현재) |
|---|---|---|
| **출력층** | Softmax (합=1) | Sigmoid (각 독립) |
| **Loss** | CrossEntropyLoss | BCEWithLogitsLoss |
| **중첩 감지** | 불가능 (1개만 선택) | 가능 (복수 패턴 동시 감지) |
| **라벨 형식** | 단일 정수 (0~6) | 이진 벡터 [0,1,0,1,0,0] |

### Composite Data Generation
- 2개 패턴을 **OR 연산**으로 합성 (픽셀 단위 중첩)
- C(6,2) = 15가지 조합, 조합당 60장 생성
- 학습 데이터에 중첩 패턴 포함 → 모델이 중첩 패턴 학습

---

## License

MIT License
