# FBM Defect Pattern Multi-Label Classification

Wafer chip **Fail Bit Map(FBM)** 이미지에서 불량 패턴을 감지하는 Multi-Label CNN 분류 시스템입니다.

단일 패턴뿐 아니라 **2개 이상의 불량 패턴이 중첩된 이미지**에서도 각 패턴을 독립적으로 감지할 수 있으며, **4가지 학습 전략**을 비교 평가하여 최적의 중첩 불량 분류 방법론을 도출합니다.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Defect Pattern Classes (7종)](#defect-pattern-classes-7종)
- [Experiment Design](#experiment-design)
- [Results](#results)
- [Analysis & Insights](#analysis--insights)
- [Visualization](#visualization)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [License](#license)

---

## Problem Statement

반도체 웨이퍼 FBM 이미지에서 **불량 패턴을 자동 분류**하는 것이 목표입니다.

실제 현장에서는 **2개 이상의 불량이 동시에 발생**(중첩 불량)하는 경우가 빈번하며, 기존 단일 분류 모델로는 이를 정확히 감지하기 어렵습니다.

### 핵심 질문

> **"중첩 불량을 더 잘 분류하기 위한 최적의 학습 전략은 무엇인가?"**

이를 검증하기 위해 **4가지 평가 조건**을 설계하고, **Binary Classification (각 클래스 독립 이진 분류)** 기반의 Multi-Label 접근법으로 실험을 수행했습니다.

---

## Defect Pattern Classes (7종)

| Class | 설명 | Sample |
|---|---|---|
| `row_line` | 로우 방향 라인 불량 (수평 스트라이프) | ![row_line](docs/images/sample_row_line.png) |
| `col_line` | 컬럼 방향 라인 불량 (수직 스트라이프) | ![col_line](docs/images/sample_col_line.png) |
| `corner_rect` | 모서리 사각형 불량 | ![corner_rect](docs/images/sample_corner_rect.png) |
| `nail` | 손톱/반달 형태 불량 (타원형) | ![nail](docs/images/sample_nail.png) |
| `edge` | 가장자리 불량 | ![edge](docs/images/sample_edge.png) |
| `block` | 블록 형태 불량 | ![block](docs/images/sample_block.png) |
| `scatter` | **랜덤 산포 불량** (신규 추가) | ![scatter](docs/images/sample_scatter.png) |

### 중첩 패턴 예시

| 조합 | Sample |
|---|---|
| `row_line` + `nail` | ![row_line+nail](docs/images/sample_row_line+nail.png) |
| `corner_rect` + `block` | ![corner_rect+block](docs/images/sample_corner_rect+block.png) |
| `col_line` + `scatter` | ![col_line+scatter](docs/images/sample_col_line+scatter.png) |
| `edge` + `scatter` | ![edge+scatter](docs/images/sample_edge+scatter.png) |

> **FBM Image Size**: 128 x 46 pixels (Grayscale)
>
> **Scatter 패턴**: 5~15% 밀도의 랜덤 산포 + 소규모 클러스터로 구성된 랜덤성 결함

---

## Experiment Design

### 평가 Metrics

| Metric | 정의 | 의미 |
|---|---|---|
| **Subset Accuracy** (Exact Match) | 7개 레이블 벡터가 **완전히 일치**하는 비율 | 엄격한 기준 — 하나라도 틀리면 오답 |
| **Hamming Accuracy** | 개별 레이블의 정확도 평균 (1 - Hamming Loss) | 유연한 기준 — 레이블 단위 정확도 |

### 4가지 평가 조건

| 평가 | 학습 데이터 | 모델 | 핵심 의도 |
|---|---|---|---|
| **Eval 1** | 단일 패턴만 (2,040장) | CNN Classifier | Baseline — 단일만 보고 중첩을 맞출 수 있는가? |
| **Eval 2** | 단일 + 합성 중첩 (2,922장) | CNN Classifier | 합성 중첩 데이터의 효과 검증 |
| **Eval 3** | 단일 + 합성 중첩 + **마스킹** (2,922장) | CNN Classifier | Occlusion 내성 향상 (Cutout Augmentation) |
| **Eval 4** | Eval 1~3 중 최적 조건의 데이터 | **Spatial Attention Detection** 모델 | Object Detection 기법 적용 효과 |

### 공통 조건

| 항목 | 값 |
|---|---|
| Epochs | 30 |
| Batch Size | 32 |
| Learning Rate | 0.001 (StepLR: step=10, gamma=0.5) |
| Loss | BCEWithLogitsLoss |
| Optimizer | Adam |
| Threshold | 0.5 |
| GPU | NVIDIA RTX 3080 |
| Mixed Precision | FP16 (AMP) |

### 테스트셋 구성

| 테스트셋 | 이미지 수 | 구성 |
|---|---|---|
| **test_single** | 360장 | 단일 패턴 7종 + 정상 |
| **test_composite** | 378장 | 2-패턴 중첩 (C(7,2)=21 조합) |

---

## Results

### 종합 성능 비교

| 평가 조건 | 단일 불량<br>Subset Acc | 단일 불량<br>Hamming Acc | 중첩 불량<br>Subset Acc | 중첩 불량<br>Hamming Acc |
|---|:---:|:---:|:---:|:---:|
| **Eval 1** (단일학습) | 99.2% | 99.9% | **23.8%** | 88.9% |
| **Eval 2** (합성중첩) | 99.2% | 99.8% | 86.8% | 98.0% |
| **Eval 3** (마스킹) | 99.2% | 99.8% | 85.2% | 97.8% |
| **Eval 4** (Detection) | **99.7%** | **100.0%** | **88.6%** | **98.2%** |

![Overall Comparison](docs/images/01_overall_comparison.png)

### 클래스별 정확도 (중첩 불량 테스트)

| 클래스 | Eval1 | Eval2 | Eval3 | Eval4 |
|---|:---:|:---:|:---:|:---:|
| `row_line` | 89.9% | 98.1% | 98.7% | **98.9%** |
| `col_line` | 89.2% | 97.1% | 97.6% | **98.9%** |
| `corner_rect` | 84.1% | **96.0%** | 95.5% | 95.5% |
| `nail` | 85.2% | **98.1%** | **98.1%** | 97.1% |
| `edge` | 83.1% | 97.4% | 96.6% | **97.6%** |
| `block` | 92.1% | 99.2% | 97.9% | **99.5%** |
| `scatter` | 98.9% | **100.0%** | **100.0%** | **100.0%** |

![Per-Class Heatmap](docs/images/02_per_class_heatmap.png)

### 클래스별 F1 Score (중첩 불량 테스트)

| 클래스 | Eval1 | Eval2 | Eval3 | Eval4 |
|---|:---:|:---:|:---:|:---:|
| `row_line` | 0.787 | 0.967 | 0.976 | **0.981** |
| `col_line` | 0.766 | 0.946 | 0.957 | **0.981** |
| `corner_rect` | 0.620 | **0.930** | 0.920 | 0.916 |
| `nail` | 0.654 | **0.967** | **0.967** | 0.947 |
| `edge` | 0.579 | 0.952 | 0.936 | **0.958** |
| `block` | 0.839 | 0.986 | 0.962 | **0.991** |
| `scatter` | 0.981 | **1.000** | **1.000** | **1.000** |

---

## Analysis & Insights

### 1. 단일 패턴만으로는 중첩 불량을 분류할 수 없다 (Eval 1)

- 단일 불량 분류: **99.2%** (매우 우수)
- 중첩 불량 분류: **23.8%** (사실상 실패)

> **핵심 원인**: Precision은 높지만 (98~100%) **Recall이 40~65%로 극도로 낮음**
>
> → 모델이 중첩 이미지에서 "하나의 패턴만 감지"하고 나머지를 놓침
> → 특히 `edge`(41%), `corner_rect`(45%)는 다른 패턴에 가려져 감지 실패

**결론**: 단일 패턴 학습만으로는 중첩 상황에서의 **패턴 공존(co-occurrence)** 을 학습하지 못함.

### 2. 합성 중첩 데이터가 가장 큰 성능 향상을 가져온다 (Eval 2)

- 중첩 Subset Accuracy: 23.8% → **86.8%** (**+63.0%p**)
- 중첩 Hamming Accuracy: 88.9% → **98.0%** (+9.1%p)

> **핵심 요인**: 단순 OR 합성 방식으로 중첩 이미지를 만들어 학습에 포함한 것만으로도 **Recall이 모든 클래스에서 90%+ 수준으로 향상**
>
> → 모델이 "두 패턴이 동시에 존재할 수 있다"는 것을 학습
> → 가장 비용 대비 효과가 큰 전략

**결론**: 실제 중첩 데이터가 부족하더라도, 합성 데이터로 대체 가능. **데이터 합성이 핵심 전략.**

### 3. 마스킹 학습은 이 데이터셋에서 추가 효과가 제한적이다 (Eval 3)

- 중첩 Subset Accuracy: 86.8% (Eval2) → **85.2%** (Eval3) (-1.6%p)

> **분석**: Cutout 마스킹이 occlusion 내성을 높이려는 의도였으나, 오히려 소폭 하락
>
> **원인 추정**:
> 1. FBM 이미지가 128x46으로 작아, 마스킹 영역이 상대적으로 너무 넓어 핵심 특징 파괴
> 2. FBM의 바이너리 특성상, 일반 자연 이미지 대비 마스킹 정규화의 효과가 작음
> 3. Eval2에서 이미 합성 데이터로 충분한 occlusion 내성을 확보

**결론**: 마스킹 전략은 이미지 크기와 특성에 따라 **튜닝이 필요**하며, 합성 중첩 데이터가 이미 occlusion 학습을 대체하고 있음.

### 4. Spatial Attention Detection 모델이 최고 성능 달성 (Eval 4)

- 중첩 Subset Accuracy: **88.6%** (전체 최고)
- 단일 Subset Accuracy: **99.7%** (전체 최고)

> **핵심 차이점**:
> - 기존 CNN: Global Average Pooling → 공간 정보 소실 → 중첩 시 특징 혼합
> - Detection 모델: 클래스별 Spatial Attention Map → 각 결함 위치를 독립 탐지

| 구조 | 파라미터 | 특징 |
|---|---|---|
| CNN Classifier | 406K | AdaptiveAvgPool → 공간 정보 압축 |
| Detection Model | 620K | 클래스별 Attention Head → 공간 독립 탐지 |

> Attention Map 시각화 결과, 모델이 각 클래스에 해당하는 **서로 다른 공간 영역에 집중**하는 것을 확인:
> - `row_line` → 수평 라인 영역에 높은 attention
> - `nail` → 타원형 영역에 집중
> - `scatter` → 이미지 전반에 균일한 attention

**결론**: Spatial Attention 기반 모델은 중첩 불량에서 **공간적으로 분리된 결함을 독립적으로 탐지**할 수 있어, 표준 CNN 대비 우수.

### 5. Scatter 패턴의 고유한 특성

- 모든 평가 조건에서 **98.9~100%** 정확도
- Eval1(단일 학습)에서도 98.9%로 거의 완벽

> **원인**: 랜덤 산포 패턴은 이미지 전체에 분산되어 있어, 다른 패턴과 **구조적으로 구별**됨
>
> → 다른 패턴(구조적 형태)과 겹쳐도 scatter의 랜덤 텍스처는 보존됨
> → CNN이 전역 통계(global statistics)로 쉽게 감지 가능

### 종합 성능 순위 (중첩 불량 기준)

```
Eval4 (Detection)   ████████████████████████████████████████████  88.6%
Eval2 (합성 중첩)    ███████████████████████████████████████████   86.8%
Eval3 (마스킹)       ██████████████████████████████████████████    85.2%
Eval1 (단일 학습)    ███████████                                   23.8%
```

---

## Visualization

### 1. 종합 성능 비교 & 학습 곡선

4개 평가 조건의 Subset/Hamming Accuracy 비교 및 Training/Validation Loss 추이.

![Overall Comparison](docs/images/01_overall_comparison.png)

### 2. 클래스별 Accuracy Heatmap

단일/중첩 테스트에서 각 클래스의 정확도를 히트맵으로 표현.

![Per-Class Heatmap](docs/images/02_per_class_heatmap.png)

### 3. F1 Score Radar Chart

클래스별 F1 Score를 레이더 차트로 비교. Eval1의 중첩 불량 F1이 전반적으로 낮은 반면, Eval2~4는 균형 잡힌 성능을 보임.

![F1 Radar Chart](docs/images/03_f1_radar_chart.png)

### 4. Precision & Recall 비교

Eval1의 **Recall 부족** 문제가 가장 뚜렷하게 드러나는 차트. 합성 데이터 추가 후 Recall이 대폭 개선됨.

![Precision Recall](docs/images/04_precision_recall.png)

### 5. 샘플 예측 결과

중첩 테스트 이미지에 대한 각 평가 조건의 실제 예측 결과. 초록색=정답, 빨간색=오답.

![Sample Predictions](docs/images/05_sample_predictions.png)

### 6. Detection Model Attention Maps

Eval4 모델의 Spatial Attention Map. 각 클래스가 이미지의 다른 공간 영역에 집중하는 것을 확인.

![Detection Attention Maps](docs/images/06_detection_attention_maps.png)

---

## Model Architecture

### Standard CNN Classifier (Eval 1~3)

```
Input: 1 x 46 x 128 (Grayscale FBM)
  │
  ├── Conv2d(1→32) + BN + ReLU + MaxPool2d(2)     → 32 x 23 x 64
  ├── Conv2d(32→64) + BN + ReLU + MaxPool2d(2)    → 64 x 11 x 32
  ├── Conv2d(64→128) + BN + ReLU + MaxPool2d(2)   → 128 x 5 x 16
  ├── Conv2d(128→256) + BN + ReLU + AdaptiveAvgPool(1) → 256 x 1 x 1
  │
  ├── Dropout(0.3)
  └── Linear(256→7)  ← raw logits

Output: 7 logits → Sigmoid → Threshold(0.5)
Parameters: ~406K
```

### Spatial Attention Detection Model (Eval 4)

```
Input: 1 x 46 x 128
  │
  ├── Backbone (Shared Feature Extraction)
  │   ├── Conv(1→32) + BN + ReLU + MaxPool    → 32 x 23 x 64
  │   ├── Conv(32→64) + BN + ReLU + MaxPool   → 64 x 11 x 32
  │   ├── Conv(64→128) + BN + ReLU + MaxPool  → 128 x 5 x 16
  │   └── Conv(128→256) + BN + ReLU           → 256 x 5 x 16
  │
  ├── Per-Class Spatial Attention Heads (×7)
  │   ├── Conv1x1(256→64) + ReLU + Conv1x1(64→1) + Sigmoid
  │   └── → Attention Map (1 x 5 x 16) per class
  │
  ├── Attended Feature Aggregation
  │   └── (feature × attention).mean(spatial) → 256-dim per class
  │
  └── Per-Class FC: Linear(256→64) + ReLU + Dropout + Linear(64→1)

Output: 7 logits → Sigmoid → Threshold(0.5)
Parameters: ~620K
```

**핵심 차이**: Standard CNN은 `AdaptiveAvgPool`로 공간 정보를 완전히 압축하지만, Detection 모델은 **클래스별 독립 Attention Map**으로 각 결함의 공간 위치를 보존하여 탐지합니다.

---

## Project Structure

```
fbm_classification/
├── README.md                    # 프로젝트 문서 (본 파일)
├── requirements.txt             # 의존성 패키지
├── fbm_model.py                 # CNN 모델 정의 (FBMClassifier)
├── generate_fbm_data.py         # 합성 FBM 데이터 생성 (단일 + 중첩)
├── train.py                     # 기본 학습 + 평가 스크립트
├── run_evaluation.py            # ★ 종합 평가 파이프라인 (Eval 1~4)
├── detect.py                    # CLI 추론 스크립트
├── webcam_detect.py             # GUI(tkinter) 추론 애플리케이션
├── generate_sample_images.py    # README용 샘플 이미지 생성
│
├── docs/images/                 # README 시각화 이미지
│   ├── 01_overall_comparison.png
│   ├── 02_per_class_heatmap.png
│   ├── 03_f1_radar_chart.png
│   ├── 04_precision_recall.png
│   ├── 05_sample_predictions.png
│   ├── 06_detection_attention_maps.png
│   └── sample_*.png            # 패턴 샘플 이미지
│
├── data/eval_dataset/           # (생성됨) 평가용 데이터셋
│   ├── train_single/            # Eval1 학습 데이터
│   ├── train_composite/         # Eval2~4 학습 데이터
│   ├── test_single/             # 단일 패턴 테스트셋
│   └── test_composite/          # 중첩 패턴 테스트셋
│
└── runs/evaluation/             # (생성됨) 평가 결과
    ├── eval1_best.pt ~ eval4_best.pt
    └── visualizations/          # 시각화 결과
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

> NVIDIA GPU 사용 시:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 2. Run Full Evaluation Pipeline (권장)

```bash
python run_evaluation.py
```

이 스크립트 하나로 **데이터 생성 → 4가지 평가 학습/테스트 → 시각화**가 자동으로 수행됩니다.

옵션:
```bash
python run_evaluation.py --epochs 50 --count 500 --batch-size 64
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--epochs` | 30 | 학습 에포크 수 |
| `--count` | 300 | 클래스당 이미지 수 |
| `--batch-size` | 32 | 배치 크기 |
| `--lr` | 0.001 | 학습률 |
| `--seed` | 42 | 랜덤 시드 |

### 3. Basic Training (단일 학습만)

```bash
python generate_fbm_data.py   # 데이터 생성
python train.py --epochs 30   # 학습
```

### 4. Inference

```bash
# CLI
python detect.py --source path/to/image.png

# GUI
python webcam_detect.py
```

---

## Key Design Decisions

### Multi-Label vs Multi-Class

| | Multi-Class (기존) | Multi-Label (본 프로젝트) |
|---|---|---|
| 출력층 | Softmax (합=1) | Sigmoid (각 독립) |
| Loss | CrossEntropyLoss | BCEWithLogitsLoss |
| 중첩 감지 | 불가능 | **가능** |
| 라벨 형식 | 단일 정수 (0~7) | 이진 벡터 `[0,1,0,1,0,0,1]` |

### 합성 데이터 전략

- 2개 패턴을 **OR 연산**으로 합성 (픽셀 단위 중첩)
- C(7,2) = 21가지 조합 × 60장
- 합성만으로도 **+63%p Subset Accuracy 향상** 달성

### Masking Augmentation (Eval 3)

- **Cutout** 기법 적용: 이미지의 10~30% 영역을 랜덤으로 마스킹
- 70% 확률로 1~3개 마스크 적용
- 부분 정보만으로 패턴 인식 → occlusion 내성 강화 의도

### Spatial Attention Detection (Eval 4)

- 기존 GAP 대신 **클래스별 독립 Spatial Attention** 적용
- 각 결함 유형의 공간 위치를 독립적으로 탐지
- Attention Map으로 모델의 판단 근거를 시각적으로 해석 가능

---

## Environment

| 항목 | 값 |
|---|---|
| GPU | NVIDIA GeForce RTX 3080 (10GB VRAM) |
| CUDA | 12.1 |
| PyTorch | 2.5.1+cu121 |
| Python | 3.x |
| Training Time | ~2분 (4개 모델 × 30 epochs) |

---

## License

MIT License
