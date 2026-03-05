"""
FBM 불량 패턴 분류 - 종합 성능 평가 파이프라인

평가 1: 단일 패턴만 학습 → 중첩 불량 분류
평가 2: 합성 중첩 이미지 학습 → 중첩 불량 분류
평가 3: 마스킹 학습 (occlusion 개선) → 중첩 불량 분류
평가 4: Object Detection 모델 (최적 조건) → 중첩 불량 분류

Metrics: Subset Accuracy (Exact Match), Hamming Accuracy
Binary Classification Task (각 클래스 독립 이진 분류)

사용법:
    python run_evaluation.py
    python run_evaluation.py --epochs 30 --count 300
"""

import argparse
import csv
import json
import random
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# ── 기존 모듈 임포트 ──
from generate_fbm_data import (
    FBM_H, FBM_W, DEFECT_CLASSES, PATTERN_GENERATORS, PATTERN_NAMES_KR,
    generate_composite_fbm, add_noise, fbm_to_image, make_label_vector,
)
from fbm_model import FBMClassifier

NUM_CLASSES = len(DEFECT_CLASSES)

# ============================================================
#  Dataset Classes
# ============================================================

class FBMDataset(Dataset):
    """Multi-Label FBM 데이터셋"""

    def __init__(self, data_dir, augment=False):
        self.data_dir = Path(data_dir)
        self.augment = augment
        self.samples = []
        self.class_names = []

        csv_path = self.data_dir / "labels.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"labels.csv 없음: {csv_path}")

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            self.class_names = header[1:]
            for row in reader:
                fname = row[0]
                label = [int(x) for x in row[1:]]
                img_path = self.data_dir / "images" / fname
                if img_path.exists():
                    self.samples.append((str(img_path), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0

        if self.augment:
            if np.random.random() > 0.5:
                arr = np.fliplr(arr).copy()
            if np.random.random() > 0.5:
                arr = np.flipud(arr).copy()
            if np.random.random() > 0.5:
                noise = np.random.normal(0, 0.02, arr.shape).astype(np.float32)
                arr = np.clip(arr + noise, 0, 1)

        tensor = torch.from_numpy(arr).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return tensor, label_tensor


class MaskedFBMDataset(FBMDataset):
    """Masking Augmentation 적용 데이터셋 (Eval 3: Occlusion 개선)"""

    def __getitem__(self, idx):
        tensor, label = super().__getitem__(idx)

        if self.augment and random.random() < 0.7:
            _, h, w = tensor.shape
            num_masks = random.randint(1, 3)
            for _ in range(num_masks):
                mask_h = random.randint(h // 6, h // 3)
                mask_w = random.randint(w // 6, w // 3)
                y0 = random.randint(0, h - mask_h)
                x0 = random.randint(0, w - mask_w)
                tensor[:, y0:y0 + mask_h, x0:x0 + mask_w] = 0

        return tensor, label


# ============================================================
#  Detection Model (Eval 4)
# ============================================================

class FBMDetectionClassifier(nn.Module):
    """
    Object Detection 기반 분류 모델

    Spatial Attention 기법을 활용하여 각 클래스별로 독립적인
    공간 주의 맵(heatmap)을 생성하고, 해당 영역의 특징을 기반으로 분류.
    - 기존 CNN: Global Average Pooling → 공간 정보 소실 → 중첩 시 혼동
    - Detection 모델: 클래스별 공간 맵 → 개별 결함 위치 독립 탐지
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes

        # Backbone (공간 정보 보존)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        # 클래스별 공간 주의(attention) 헤드
        self.attn_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 64, 1), nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 1), nn.Sigmoid()
            ) for _ in range(num_classes)
        ])

        # 클래스별 분류 FC
        self.class_fcs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 64), nn.ReLU(inplace=True), nn.Dropout(0.3),
                nn.Linear(64, 1)
            ) for _ in range(num_classes)
        ])

    def forward(self, x):
        f = self.conv1(x)
        f = self.conv2(f)
        f = self.conv3(f)
        f = self.conv4(f)   # (B, 256, H/8, W/8)

        outputs = []
        self._attn_maps = []  # 시각화용 저장

        for i in range(self.num_classes):
            attn = self.attn_heads[i](f)            # (B, 1, H', W')
            self._attn_maps.append(attn.detach())
            attended = (f * attn).mean(dim=[2, 3])   # (B, 256)
            out = self.class_fcs[i](attended)        # (B, 1)
            outputs.append(out)

        return torch.cat(outputs, dim=1)  # (B, num_classes) logits


# ============================================================
#  Metrics
# ============================================================

def compute_metrics(preds: torch.Tensor, labels: torch.Tensor):
    """
    Binary classification metrics 계산

    Args:
        preds: (N, C) binary predictions
        labels: (N, C) binary ground truth

    Returns:
        dict with subset_accuracy, hamming_accuracy, per_class_accuracy,
             per_class_precision, per_class_recall, per_class_f1
    """
    n, c = labels.shape

    # Subset Accuracy (Exact Match): 모든 레이블이 정확히 일치하는 비율
    subset_acc = (preds == labels).all(dim=1).float().mean().item()

    # Hamming Accuracy: 1 - Hamming Loss = 개별 레이블 정확도 평균
    hamming_acc = (preds == labels).float().mean().item()

    # Per-class metrics
    per_class_acc = (preds == labels).float().mean(dim=0)

    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []

    for i in range(c):
        tp = ((preds[:, i] == 1) & (labels[:, i] == 1)).sum().float()
        fp = ((preds[:, i] == 1) & (labels[:, i] == 0)).sum().float()
        fn = ((preds[:, i] == 0) & (labels[:, i] == 1)).sum().float()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        per_class_precision.append(precision.item())
        per_class_recall.append(recall.item())
        per_class_f1.append(f1.item())

    return {
        "subset_accuracy": subset_acc,
        "hamming_accuracy": hamming_acc,
        "per_class_accuracy": per_class_acc.tolist(),
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
    }


# ============================================================
#  Data Generation
# ============================================================

def generate_evaluation_data(output_dir, count_per_class=300, seed=42):
    """평가용 데이터셋 생성 (공유 테스트셋 + 평가별 훈련 데이터)"""
    random.seed(seed)
    np.random.seed(seed)

    base = Path(output_dir)
    all_combos = list(combinations(DEFECT_CLASSES, 2))
    combo_count = max(30, count_per_class // 5)

    # 디렉토리 구조
    splits = {
        "train_single": base / "train_single",       # Eval 1 학습용
        "train_composite": base / "train_composite",  # Eval 2,3,4 학습용
        "test_single": base / "test_single",          # 단일 패턴 테스트
        "test_composite": base / "test_composite",    # 중첩 패턴 테스트
    }

    for p in splits.values():
        (p / "images").mkdir(parents=True, exist_ok=True)

    rows = {k: [] for k in splits}
    header = ["filename"] + DEFECT_CLASSES

    print("=" * 60)
    print("  평가용 데이터셋 생성")
    print("=" * 60)
    print(f"  FBM 크기: {FBM_W} x {FBM_H}")
    print(f"  클래스: {DEFECT_CLASSES}")
    print(f"  클래스당 이미지: {count_per_class}장")
    print()

    train_ratio = 0.7
    test_ratio = 0.15  # 나머지 0.15는 validation 대신 test로

    # ── 1) Normal ──
    print(f"  생성 중: {'normal':20s} ... ", end="", flush=True)
    for i in range(count_per_class):
        fbm = np.zeros((FBM_H, FBM_W), dtype=np.uint8)
        fbm = add_noise(fbm, random.uniform(0.001, 0.01))
        fname = f"normal_{i:04d}.png"
        label = make_label_vector([])

        if i < int(count_per_class * train_ratio):
            split_key = "train_single"
        elif i < int(count_per_class * (train_ratio + test_ratio)):
            split_key = "test_single"
        else:
            split_key = "train_single"  # 나머지도 학습용

        fbm_to_image(fbm).save(splits[split_key] / "images" / fname)
        rows[split_key].append([fname] + label)
    print(f"{count_per_class}장")

    # ── 2) 단일 패턴 ──
    for pattern_name, gen_func in PATTERN_GENERATORS.items():
        kr = PATTERN_NAMES_KR[pattern_name]
        print(f"  생성 중: {pattern_name:20s} ({kr}) ... ", end="", flush=True)
        label = make_label_vector([pattern_name])

        for i in range(count_per_class):
            fbm = add_noise(gen_func(), 0.003)
            fname = f"single_{pattern_name}_{i:04d}.png"

            if i < int(count_per_class * train_ratio):
                split_key = "train_single"
            elif i < int(count_per_class * (train_ratio + test_ratio)):
                split_key = "test_single"
            else:
                split_key = "train_single"

            fbm_to_image(fbm).save(splits[split_key] / "images" / fname)
            rows[split_key].append([fname] + label)
        print(f"{count_per_class}장")

    # ── 3) train_single 데이터를 train_composite에도 복사 ──
    print(f"\n  train_composite에 단일패턴 복사 중...")
    import shutil
    for fname_label in rows["train_single"]:
        fname = fname_label[0]
        src = splits["train_single"] / "images" / fname
        dst = splits["train_composite"] / "images" / fname
        if src.exists():
            shutil.copy2(str(src), str(dst))
    rows["train_composite"].extend(rows["train_single"].copy())

    # ── 4) 중첩 패턴 (2개 조합) ──
    print()
    for c1, c2 in all_combos:
        combo_name = f"{c1}+{c2}"
        print(f"  생성 중: {combo_name:30s} (중첩) ... ", end="", flush=True)
        label = make_label_vector([c1, c2])

        for i in range(combo_count):
            fbm = generate_composite_fbm([c1, c2])
            fname = f"combo_{c1}+{c2}_{i:04d}.png"

            if i < int(combo_count * 0.70):
                # train_composite에만 추가 (Eval 2,3,4 학습용)
                fbm_to_image(fbm).save(splits["train_composite"] / "images" / fname)
                rows["train_composite"].append([fname] + label)
            else:
                # 테스트용
                fbm_to_image(fbm).save(splits["test_composite"] / "images" / fname)
                rows["test_composite"].append([fname] + label)
        print(f"{combo_count}장")

    # ── CSV 저장 ──
    for key, split_rows in rows.items():
        csv_path = splits[key] / "labels.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(split_rows)

    (base / "classes.txt").write_text("\n".join(DEFECT_CLASSES), encoding="utf-8")

    print(f"\n  Train(단일만):    {len(rows['train_single'])}장")
    print(f"  Train(단일+중첩): {len(rows['train_composite'])}장")
    print(f"  Test(단일):       {len(rows['test_single'])}장")
    print(f"  Test(중첩):       {len(rows['test_composite'])}장")
    print("=" * 60)

    return {k: str(v) for k, v in splits.items()}


# ============================================================
#  Training
# ============================================================

def train_model(model, train_loader, val_loader, epochs, lr, device, use_cuda, label=""):
    """모델 학습 (공통 함수)"""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    history = {"train_loss": [], "val_loss": [], "val_subset_acc": [], "val_hamming_acc": []}
    best_val_metric = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss = 0.0
        train_n = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_cuda):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * inputs.size(0)
            train_n += inputs.size(0)
        scheduler.step()
        train_loss /= train_n

        # Validation
        val_loss, val_metrics = evaluate_model(model, val_loader, criterion, device, use_cuda)

        et = time.time() - t0
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_subset_acc"].append(val_metrics["subset_accuracy"])
        history["val_hamming_acc"].append(val_metrics["hamming_accuracy"])

        marker = ""
        metric = val_metrics["subset_accuracy"]
        if metric > best_val_metric:
            best_val_metric = metric
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker = " *"

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"    [{label}] Ep {epoch:3d}/{epochs}"
                f"  TrLoss:{train_loss:.4f}"
                f"  VaLoss:{val_loss:.4f}"
                f"  SubsetAcc:{val_metrics['subset_accuracy']:.1%}"
                f"  HammAcc:{val_metrics['hamming_accuracy']:.1%}"
                f"  ({et:.1f}s){marker}"
            )

    if best_state:
        model.load_state_dict(best_state)

    return model, history, best_val_metric


def evaluate_model(model, loader, criterion, device, use_cuda, threshold=0.5):
    """모델 평가 → (loss, metrics_dict)"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_cuda):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            probs = torch.sigmoid(outputs)
            preds = (probs >= threshold).float()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)
    avg_loss = total_loss / all_labels.size(0)

    metrics = compute_metrics(all_preds, all_labels)
    metrics["all_preds"] = all_preds
    metrics["all_labels"] = all_labels
    metrics["all_probs"] = all_probs

    return avg_loss, metrics


# ============================================================
#  Visualization
# ============================================================

def create_visualizations(all_results, output_dir, splits):
    """종합 시각화 생성"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    # 한글 폰트 설정 시도
    try:
        rcParams["font.family"] = "Malgun Gothic"
    except Exception:
        pass
    rcParams["axes.unicode_minus"] = False

    viz_dir = Path(output_dir) / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    eval_names = ["Eval1\n(단일학습)", "Eval2\n(합성중첩)", "Eval3\n(마스킹)", "Eval4\n(Detection)"]
    eval_keys = ["eval1", "eval2", "eval3", "eval4"]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    # ── 1) 종합 성능 비교 Bar Chart ──
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle("FBM Defect Classification - Evaluation Results", fontsize=28, fontweight="bold")

    # 1a) Subset Accuracy 비교
    ax = axes[0, 0]
    for test_type, ax_idx in [("single", axes[0, 0]), ("composite", axes[0, 1])]:
        subset_accs = []
        hamming_accs = []
        for ek in eval_keys:
            r = all_results[ek].get(f"test_{test_type}", {})
            subset_accs.append(r.get("subset_accuracy", 0))
            hamming_accs.append(r.get("hamming_accuracy", 0))

        x = np.arange(len(eval_names))
        width = 0.35
        bars1 = ax_idx.bar(x - width / 2, subset_accs, width, label="Subset Accuracy", color=colors, alpha=0.8)
        bars2 = ax_idx.bar(x + width / 2, hamming_accs, width, label="Hamming Accuracy", color=colors, alpha=0.5)
        ax_idx.set_xlabel("Evaluation", fontsize=16)
        ax_idx.set_ylabel("Accuracy", fontsize=16)
        test_label = "Single Defect" if test_type == "single" else "Composite Defect"
        ax_idx.set_title(f"{test_label} Test Performance", fontsize=18)
        ax_idx.set_xticks(x)
        ax_idx.set_xticklabels(eval_names, fontsize=14)
        ax_idx.legend(loc="lower right", fontsize=14)
        ax_idx.set_ylim(0, 1.05)
        ax_idx.grid(axis="y", alpha=0.3)
        ax_idx.tick_params(axis="y", labelsize=13)

        for bar in bars1:
            h = bar.get_height()
            ax_idx.text(bar.get_x() + bar.get_width() / 2., h + 0.01, f"{h:.1%}",
                       ha="center", va="bottom", fontsize=14)
        for bar in bars2:
            h = bar.get_height()
            ax_idx.text(bar.get_x() + bar.get_width() / 2., h + 0.01, f"{h:.1%}",
                       ha="center", va="bottom", fontsize=14)

    # 1b) Training Loss Curves
    ax = axes[1, 0]
    for i, ek in enumerate(eval_keys):
        hist = all_results[ek].get("history", {})
        if "train_loss" in hist:
            ax.plot(hist["train_loss"], label=f"Eval{i+1} Train", color=colors[i], linestyle="-")
        if "val_loss" in hist:
            ax.plot(hist["val_loss"], label=f"Eval{i+1} Val", color=colors[i], linestyle="--")
    ax.set_xlabel("Epoch", fontsize=16)
    ax.set_ylabel("Loss", fontsize=16)
    ax.set_title("Training & Validation Loss", fontsize=18)
    ax.legend(fontsize=12, ncol=2)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=13)

    # 1c) Validation Subset Accuracy Curves
    ax = axes[1, 1]
    for i, ek in enumerate(eval_keys):
        hist = all_results[ek].get("history", {})
        if "val_subset_acc" in hist:
            ax.plot(hist["val_subset_acc"], label=f"Eval{i+1} Subset", color=colors[i], linestyle="-")
        if "val_hamming_acc" in hist:
            ax.plot(hist["val_hamming_acc"], label=f"Eval{i+1} Hamming", color=colors[i], linestyle="--")
    ax.set_xlabel("Epoch", fontsize=16)
    ax.set_ylabel("Accuracy", fontsize=16)
    ax.set_title("Validation Accuracy Progress", fontsize=18)
    ax.legend(fontsize=12, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=13)

    plt.tight_layout()
    plt.savefig(viz_dir / "01_overall_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [시각화] 종합 성능 비교: {viz_dir / '01_overall_comparison.png'}")

    # ── 2) Per-Class Performance Heatmap ──
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))

    for test_idx, test_type in enumerate(["single", "composite"]):
        ax = axes[test_idx]
        data_matrix = []
        for ek in eval_keys:
            r = all_results[ek].get(f"test_{test_type}", {})
            per_class = r.get("per_class_accuracy", [0] * NUM_CLASSES)
            data_matrix.append(per_class)

        data_matrix = np.array(data_matrix)
        im = ax.imshow(data_matrix, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")

        ax.set_xticks(range(NUM_CLASSES))
        ax.set_xticklabels(DEFECT_CLASSES, rotation=45, ha="right", fontsize=16)
        ax.set_yticks(range(4))
        ax.set_yticklabels(["Eval1", "Eval2", "Eval3", "Eval4"], fontsize=16)

        for i in range(4):
            for j in range(NUM_CLASSES):
                val = data_matrix[i, j]
                color = "white" if val < 0.75 else "black"
                ax.text(j, i, f"{val:.1%}", ha="center", va="center", fontsize=16, color=color, fontweight="bold")

        test_label = "Single Defect" if test_type == "single" else "Composite Defect"
        ax.set_title(f"Per-Class Accuracy ({test_label})", fontsize=18)
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle("Per-Class Accuracy Heatmap", fontsize=28, fontweight="bold")
    plt.tight_layout()
    plt.savefig(viz_dir / "02_per_class_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [시각화] 클래스별 성능 히트맵: {viz_dir / '02_per_class_heatmap.png'}")

    # ── 3) Per-Class F1 Score Radar Chart ──
    fig, axes = plt.subplots(1, 2, figsize=(22, 10), subplot_kw=dict(polar=True))

    for test_idx, test_type in enumerate(["single", "composite"]):
        ax = axes[test_idx]
        angles = np.linspace(0, 2 * np.pi, NUM_CLASSES, endpoint=False).tolist()
        angles += angles[:1]

        for i, ek in enumerate(eval_keys):
            r = all_results[ek].get(f"test_{test_type}", {})
            f1_scores = r.get("per_class_f1", [0] * NUM_CLASSES)
            values = f1_scores + f1_scores[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=f"Eval{i+1}", color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(DEFECT_CLASSES, fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_title(f"{'Single' if test_type == 'single' else 'Composite'} - Per-Class F1", pad=20, fontsize=18)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=14)

    plt.suptitle("Per-Class F1 Score Radar Chart", fontsize=28, fontweight="bold")
    plt.tight_layout()
    plt.savefig(viz_dir / "03_f1_radar_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [시각화] F1 레이더 차트: {viz_dir / '03_f1_radar_chart.png'}")

    # ── 4) Precision / Recall 비교 ──
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))

    for test_idx, test_type in enumerate(["single", "composite"]):
        for metric_idx, metric_name in enumerate(["per_class_precision", "per_class_recall"]):
            ax = axes[test_idx, metric_idx]
            x = np.arange(NUM_CLASSES)
            width = 0.18

            for i, ek in enumerate(eval_keys):
                r = all_results[ek].get(f"test_{test_type}", {})
                values = r.get(metric_name, [0] * NUM_CLASSES)
                ax.bar(x + i * width, values, width, label=f"Eval{i+1}", color=colors[i], alpha=0.8)

            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels(DEFECT_CLASSES, rotation=45, ha="right", fontsize=14)
            ax.set_ylim(0, 1.1)
            ax.grid(axis="y", alpha=0.3)
            ax.tick_params(axis="y", labelsize=13)

            metric_label = "Precision" if "precision" in metric_name else "Recall"
            test_label = "Single" if test_type == "single" else "Composite"
            ax.set_title(f"{test_label} - {metric_label}", fontsize=18)
            ax.set_ylabel(metric_label, fontsize=14)
            ax.legend(fontsize=14)

    plt.suptitle("Precision & Recall by Class", fontsize=28, fontweight="bold")
    plt.tight_layout()
    plt.savefig(viz_dir / "04_precision_recall.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [시각화] Precision/Recall: {viz_dir / '04_precision_recall.png'}")

    # ── 5) Sample Predictions 시각화 (Composite Test) ──
    _visualize_sample_predictions(all_results, splits, viz_dir)

    # ── 6) 결과 요약 테이블 저장 ──
    _save_results_table(all_results, viz_dir)

    print(f"\n  모든 시각화 저장 완료: {viz_dir}")


def _visualize_sample_predictions(all_results, splits, viz_dir):
    """중첩 패턴 테스트 이미지에 대한 예측 결과 시각화"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    test_dir = Path(splits["test_composite"])
    if not (test_dir / "labels.csv").exists():
        return

    # 샘플 이미지 로드
    dataset = FBMDataset(test_dir, augment=False)
    if len(dataset) < 8:
        return

    n_samples = min(8, len(dataset))
    indices = random.sample(range(len(dataset)), n_samples)

    eval_keys = ["eval1", "eval2", "eval3", "eval4"]
    fig, axes = plt.subplots(n_samples, 5, figsize=(26, n_samples * 3.5))

    for row, idx in enumerate(indices):
        tensor, label = dataset[idx]
        img = tensor.squeeze().numpy()

        # 이미지 표시
        axes[row, 0].imshow(img, cmap="hot", aspect="auto")
        true_classes = [DEFECT_CLASSES[i] for i in range(NUM_CLASSES) if label[i] == 1]
        axes[row, 0].set_title(f"True: {'+'.join(true_classes)}", fontsize=14, fontweight="bold")
        axes[row, 0].axis("off")

        # 각 Eval 결과
        for col, ek in enumerate(eval_keys):
            ax = axes[row, col + 1]
            r = all_results[ek].get("test_composite", {})
            all_preds = r.get("all_preds", None)
            all_probs = r.get("all_probs", None)

            if all_preds is not None and idx < all_preds.size(0):
                pred = all_preds[idx]
                prob = all_probs[idx] if all_probs is not None else pred
                pred_classes = [DEFECT_CLASSES[i] for i in range(NUM_CLASSES) if pred[i] == 1]

                # 확률 바 차트
                bar_colors = ["#4CAF50" if pred[i] == label[i] else "#F44336" for i in range(NUM_CLASSES)]
                ax.barh(range(NUM_CLASSES), prob.numpy(), color=bar_colors, alpha=0.7)
                ax.axvline(x=0.5, color="red", linestyle="--", linewidth=1)
                ax.set_yticks(range(NUM_CLASSES))
                ax.set_yticklabels(DEFECT_CLASSES, fontsize=12)
                ax.set_xlim(0, 1)
                ax.tick_params(axis="x", labelsize=11)
                correct = (pred == label).all().item()
                ax.set_title(f"Eval{col+1}: {'O' if correct else 'X'}", fontsize=16, fontweight="bold",
                           color="green" if correct else "red")
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=16)
                ax.set_title(f"Eval{col+1}", fontsize=16)

    plt.suptitle("Sample Predictions on Composite Test Images", fontsize=28, fontweight="bold")
    plt.tight_layout()
    plt.savefig(viz_dir / "05_sample_predictions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [시각화] 샘플 예측 결과: {viz_dir / '05_sample_predictions.png'}")


def _save_results_table(all_results, viz_dir):
    """결과 요약 테이블을 텍스트와 JSON으로 저장"""
    eval_names = ["Eval1(단일학습)", "Eval2(합성중첩)", "Eval3(마스킹)", "Eval4(Detection)"]
    eval_keys = ["eval1", "eval2", "eval3", "eval4"]

    lines = []
    lines.append("=" * 90)
    lines.append("  FBM 불량 패턴 분류 - 종합 평가 결과")
    lines.append("=" * 90)
    lines.append("")

    # Composite test results
    lines.append(f"  {'평가 조건':<22s}  {'Subset Acc':>12s}  {'Hamming Acc':>12s}  {'테스트 유형':<15s}")
    lines.append(f"  {'─' * 75}")

    for i, ek in enumerate(eval_keys):
        for test_type in ["single", "composite"]:
            r = all_results[ek].get(f"test_{test_type}", {})
            sa = r.get("subset_accuracy", 0)
            ha = r.get("hamming_accuracy", 0)
            test_label = "단일 불량" if test_type == "single" else "중첩 불량"
            lines.append(f"  {eval_names[i]:<22s}  {sa:>11.1%}  {ha:>11.1%}  {test_label:<15s}")
        lines.append("")

    lines.append("")
    lines.append("  클래스별 정확도 (중첩 불량 테스트)")
    lines.append(f"  {'─' * 75}")
    lines.append(f"  {'클래스':<16s}" + "".join(f"  {en:<16s}" for en in eval_names))

    for j, cls in enumerate(DEFECT_CLASSES):
        line = f"  {cls:<16s}"
        for ek in eval_keys:
            r = all_results[ek].get("test_composite", {})
            per_cls = r.get("per_class_accuracy", [0] * NUM_CLASSES)
            line += f"  {per_cls[j]:>14.1%}"
        lines.append(line)

    lines.append("=" * 90)

    result_text = "\n".join(lines)
    print(result_text)

    with open(viz_dir / "results_summary.txt", "w", encoding="utf-8") as f:
        f.write(result_text)

    # JSON 저장
    json_results = {}
    for i, ek in enumerate(eval_keys):
        json_results[ek] = {}
        for test_type in ["single", "composite"]:
            r = all_results[ek].get(f"test_{test_type}", {})
            json_results[ek][f"test_{test_type}"] = {
                "subset_accuracy": r.get("subset_accuracy", 0),
                "hamming_accuracy": r.get("hamming_accuracy", 0),
                "per_class_accuracy": r.get("per_class_accuracy", []),
                "per_class_precision": r.get("per_class_precision", []),
                "per_class_recall": r.get("per_class_recall", []),
                "per_class_f1": r.get("per_class_f1", []),
            }

    with open(viz_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)

    print(f"  [저장] 결과 요약: {viz_dir / 'results_summary.txt'}")
    print(f"  [저장] 결과 JSON: {viz_dir / 'results.json'}")


# ============================================================
#  Detection Model Attention Map 시각화
# ============================================================

def visualize_detection_maps(model, dataset, viz_dir, n_samples=6):
    """Eval 4 Detection 모델의 Spatial Attention Map 시각화"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    viz_dir = Path(viz_dir)
    device = next(model.parameters()).device

    if len(dataset) < n_samples:
        return

    indices = random.sample(range(len(dataset)), n_samples)
    fig, axes = plt.subplots(n_samples, NUM_CLASSES + 1, figsize=(4 * (NUM_CLASSES + 1), 4 * n_samples))

    model.eval()
    with torch.no_grad():
        for row, idx in enumerate(indices):
            tensor, label = dataset[idx]
            img = tensor.squeeze().numpy()

            # 원본 이미지
            axes[row, 0].imshow(img, cmap="hot", aspect="auto")
            true_cls = [DEFECT_CLASSES[i] for i in range(NUM_CLASSES) if label[i] == 1]
            axes[row, 0].set_title(f"True: {'+'.join(true_cls)}", fontsize=14, fontweight="bold")
            axes[row, 0].axis("off")

            # Forward pass
            inp = tensor.unsqueeze(0).to(device)
            output = model(inp)
            probs = torch.sigmoid(output).squeeze()

            # Attention maps
            if hasattr(model, "_attn_maps"):
                for c in range(NUM_CLASSES):
                    attn_map = model._attn_maps[c].squeeze().cpu().numpy()
                    ax = axes[row, c + 1]

                    # 원본 이미지 위에 attention 오버레이
                    ax.imshow(img, cmap="gray", aspect="auto", alpha=0.3)
                    im = ax.imshow(attn_map, cmap="jet", aspect="auto", alpha=0.7,
                                  vmin=0, vmax=1)
                    p = probs[c].item()
                    is_true = label[c] == 1
                    ax.set_title(f"{DEFECT_CLASSES[c]}\n{p:.2f} {'(T)' if is_true else ''}",
                               fontsize=13, fontweight="bold",
                               color="green" if is_true else "gray")
                    ax.axis("off")

    plt.suptitle("Detection Model - Spatial Attention Maps", fontsize=28, fontweight="bold")
    plt.tight_layout()
    plt.savefig(viz_dir / "06_detection_attention_maps.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [시각화] Detection Attention Maps: {viz_dir / '06_detection_attention_maps.png'}")


# ============================================================
#  Main Pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="FBM 분류 종합 평가 파이프라인")
    parser.add_argument("--epochs", type=int, default=30, help="학습 에포크 수")
    parser.add_argument("--batch-size", type=int, default=32, help="배치 크기")
    parser.add_argument("--lr", type=float, default=0.001, help="학습률")
    parser.add_argument("--count", type=int, default=300, help="클래스당 이미지 수")
    parser.add_argument("--output", type=str, default="data/eval_dataset", help="데이터셋 출력 경로")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print("\n" + "=" * 60)
    print("  FBM 불량 패턴 분류 - 종합 성능 평가")
    print("=" * 60)
    print(f"  FBM 크기: {FBM_W} x {FBM_H}")
    print(f"  클래스 ({NUM_CLASSES}개): {DEFECT_CLASSES}")
    print(f"  디바이스: {device}")
    if use_cuda:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  에포크: {args.epochs}  배치: {args.batch_size}  LR: {args.lr}")
    print()

    run_dir = Path("runs/evaluation")
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: 데이터 생성 ──
    print("\n" + "=" * 60)
    print("  Step 1: 데이터 생성")
    print("=" * 60)
    splits = generate_evaluation_data(args.output, count_per_class=args.count, seed=args.seed)

    loader_kwargs = {}
    if use_cuda:
        loader_kwargs = {"pin_memory": True, "num_workers": 4, "persistent_workers": True}
        torch.backends.cudnn.benchmark = True

    # 공통 테스트 로더
    test_single_ds = FBMDataset(splits["test_single"], augment=False)
    test_composite_ds = FBMDataset(splits["test_composite"], augment=False)
    test_single_loader = DataLoader(test_single_ds, batch_size=64, shuffle=False, **loader_kwargs)
    test_composite_loader = DataLoader(test_composite_ds, batch_size=64, shuffle=False, **loader_kwargs)
    criterion = nn.BCEWithLogitsLoss()

    all_results = {}

    # ── Step 2: 평가 1 - 단일 패턴만 학습 ──
    print("\n" + "=" * 60)
    print("  Step 2: 평가 1 - 단일 패턴만 학습")
    print("=" * 60)

    train_single_ds = FBMDataset(splits["train_single"], augment=True)
    train_single_loader = DataLoader(train_single_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)

    # validation은 test_single의 일부 사용
    val_loader_1 = DataLoader(
        FBMDataset(splits["test_single"], augment=False),
        batch_size=64, shuffle=False, **loader_kwargs
    )

    model1 = FBMClassifier(num_classes=NUM_CLASSES).to(device)
    print(f"  Train 데이터: {len(train_single_ds)}장 (단일 패턴만)")
    model1, hist1, _ = train_model(model1, train_single_loader, val_loader_1,
                                    args.epochs, args.lr, device, use_cuda, "Eval1")

    # 평가
    _, eval1_single = evaluate_model(model1, test_single_loader, criterion, device, use_cuda)
    _, eval1_composite = evaluate_model(model1, test_composite_loader, criterion, device, use_cuda)

    all_results["eval1"] = {
        "test_single": eval1_single, "test_composite": eval1_composite, "history": hist1,
    }
    print(f"\n  [Eval1] 단일 테스트 - SubsetAcc: {eval1_single['subset_accuracy']:.1%}  HammAcc: {eval1_single['hamming_accuracy']:.1%}")
    print(f"  [Eval1] 중첩 테스트 - SubsetAcc: {eval1_composite['subset_accuracy']:.1%}  HammAcc: {eval1_composite['hamming_accuracy']:.1%}")

    torch.save({"model_state_dict": model1.state_dict(), "class_names": DEFECT_CLASSES,
                "num_classes": NUM_CLASSES}, run_dir / "eval1_best.pt")

    # ── Step 3: 평가 2 - 합성 중첩 이미지 학습 ──
    print("\n" + "=" * 60)
    print("  Step 3: 평가 2 - 합성 중첩 이미지 학습")
    print("=" * 60)

    train_composite_ds = FBMDataset(splits["train_composite"], augment=True)
    train_composite_loader = DataLoader(train_composite_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)

    val_loader_2 = DataLoader(
        FBMDataset(splits["test_single"], augment=False),
        batch_size=64, shuffle=False, **loader_kwargs
    )

    model2 = FBMClassifier(num_classes=NUM_CLASSES).to(device)
    print(f"  Train 데이터: {len(train_composite_ds)}장 (단일 + 중첩)")
    model2, hist2, _ = train_model(model2, train_composite_loader, val_loader_2,
                                    args.epochs, args.lr, device, use_cuda, "Eval2")

    _, eval2_single = evaluate_model(model2, test_single_loader, criterion, device, use_cuda)
    _, eval2_composite = evaluate_model(model2, test_composite_loader, criterion, device, use_cuda)

    all_results["eval2"] = {
        "test_single": eval2_single, "test_composite": eval2_composite, "history": hist2,
    }
    print(f"\n  [Eval2] 단일 테스트 - SubsetAcc: {eval2_single['subset_accuracy']:.1%}  HammAcc: {eval2_single['hamming_accuracy']:.1%}")
    print(f"  [Eval2] 중첩 테스트 - SubsetAcc: {eval2_composite['subset_accuracy']:.1%}  HammAcc: {eval2_composite['hamming_accuracy']:.1%}")

    torch.save({"model_state_dict": model2.state_dict(), "class_names": DEFECT_CLASSES,
                "num_classes": NUM_CLASSES}, run_dir / "eval2_best.pt")

    # ── Step 4: 평가 3 - 마스킹 학습 (Occlusion 개선) ──
    print("\n" + "=" * 60)
    print("  Step 4: 평가 3 - 마스킹 학습 (Occlusion 개선)")
    print("=" * 60)

    train_masked_ds = MaskedFBMDataset(splits["train_composite"], augment=True)
    train_masked_loader = DataLoader(train_masked_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)

    val_loader_3 = DataLoader(
        FBMDataset(splits["test_single"], augment=False),
        batch_size=64, shuffle=False, **loader_kwargs
    )

    model3 = FBMClassifier(num_classes=NUM_CLASSES).to(device)
    print(f"  Train 데이터: {len(train_masked_ds)}장 (단일 + 중첩 + 마스킹)")
    model3, hist3, _ = train_model(model3, train_masked_loader, val_loader_3,
                                    args.epochs, args.lr, device, use_cuda, "Eval3")

    _, eval3_single = evaluate_model(model3, test_single_loader, criterion, device, use_cuda)
    _, eval3_composite = evaluate_model(model3, test_composite_loader, criterion, device, use_cuda)

    all_results["eval3"] = {
        "test_single": eval3_single, "test_composite": eval3_composite, "history": hist3,
    }
    print(f"\n  [Eval3] 단일 테스트 - SubsetAcc: {eval3_single['subset_accuracy']:.1%}  HammAcc: {eval3_single['hamming_accuracy']:.1%}")
    print(f"  [Eval3] 중첩 테스트 - SubsetAcc: {eval3_composite['subset_accuracy']:.1%}  HammAcc: {eval3_composite['hamming_accuracy']:.1%}")

    torch.save({"model_state_dict": model3.state_dict(), "class_names": DEFECT_CLASSES,
                "num_classes": NUM_CLASSES}, run_dir / "eval3_best.pt")

    # ── Step 5: 최적 조건 결정 + 평가 4 (Detection 모델) ──
    print("\n" + "=" * 60)
    print("  Step 5: 평가 4 - Object Detection 모델")
    print("=" * 60)

    # 최적 조건 선택 (중첩 테스트 Subset Accuracy 기준)
    composite_scores = {
        "eval1": eval1_composite["subset_accuracy"],
        "eval2": eval2_composite["subset_accuracy"],
        "eval3": eval3_composite["subset_accuracy"],
    }
    best_eval = max(composite_scores, key=composite_scores.get)
    best_score = composite_scores[best_eval]
    print(f"  최적 조건: {best_eval} (SubsetAcc: {best_score:.1%})")

    # 최적 조건의 학습 데이터 사용
    if best_eval == "eval1":
        det_train_ds_cls = FBMDataset
        det_train_path = splits["train_single"]
    elif best_eval == "eval3":
        det_train_ds_cls = MaskedFBMDataset
        det_train_path = splits["train_composite"]
    else:
        det_train_ds_cls = FBMDataset
        det_train_path = splits["train_composite"]

    det_train_ds = det_train_ds_cls(det_train_path, augment=True)
    det_train_loader = DataLoader(det_train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)

    val_loader_4 = DataLoader(
        FBMDataset(splits["test_single"], augment=False),
        batch_size=64, shuffle=False, **loader_kwargs
    )

    model4 = FBMDetectionClassifier(num_classes=NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model4.parameters())
    print(f"  Detection 모델 파라미터: {total_params:,}개")
    print(f"  Train 데이터: {len(det_train_ds)}장 ({best_eval} 조건)")

    model4, hist4, _ = train_model(model4, det_train_loader, val_loader_4,
                                    args.epochs, args.lr, device, use_cuda, "Eval4")

    _, eval4_single = evaluate_model(model4, test_single_loader, criterion, device, use_cuda)
    _, eval4_composite = evaluate_model(model4, test_composite_loader, criterion, device, use_cuda)

    all_results["eval4"] = {
        "test_single": eval4_single, "test_composite": eval4_composite, "history": hist4,
    }
    print(f"\n  [Eval4] 단일 테스트 - SubsetAcc: {eval4_single['subset_accuracy']:.1%}  HammAcc: {eval4_single['hamming_accuracy']:.1%}")
    print(f"  [Eval4] 중첩 테스트 - SubsetAcc: {eval4_composite['subset_accuracy']:.1%}  HammAcc: {eval4_composite['hamming_accuracy']:.1%}")

    torch.save({"model_state_dict": model4.state_dict(), "class_names": DEFECT_CLASSES,
                "num_classes": NUM_CLASSES}, run_dir / "eval4_best.pt")

    # ── Step 6: 시각화 ──
    print("\n" + "=" * 60)
    print("  Step 6: 결과 시각화 및 분석")
    print("=" * 60)

    create_visualizations(all_results, run_dir, splits)

    # Detection attention map 시각화
    visualize_detection_maps(model4, test_composite_ds, run_dir / "visualizations", n_samples=6)

    print("\n" + "=" * 60)
    print("  종합 평가 완료!")
    print("=" * 60)
    print(f"  결과 디렉토리: {run_dir}")
    print(f"  시각화: {run_dir / 'visualizations'}")
    print()


if __name__ == "__main__":
    main()
