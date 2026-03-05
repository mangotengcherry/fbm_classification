"""
Binary Classification Probability Distribution 시각화

각 클래스(binary classifier)별로:
- 실제 label=0 (해당 패턴 없음) 샘플의 예측 확률 분포
- 실제 label=1 (해당 패턴 있음) 샘플의 예측 확률 분포
를 히스토그램으로 시각화하여 0/1이 잘 구분되는지 확인.

분포가 잘 분리되면 → 모델이 확신을 갖고 분류
분포가 겹치면 → 해당 클래스에서 혼동 발생

사용법:
    python visualize_prob_distribution.py
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

try:
    rcParams["font.family"] = "Malgun Gothic"
except Exception:
    pass
rcParams["axes.unicode_minus"] = False

from generate_fbm_data import DEFECT_CLASSES, FBM_H, FBM_W
from fbm_model import FBMClassifier
from run_evaluation import (
    FBMDataset, MaskedFBMDataset, FBMDetectionClassifier,
    evaluate_model, NUM_CLASSES,
)


def collect_probabilities(model, loader, device, use_cuda):
    """모든 샘플의 예측 확률과 라벨 수집"""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_cuda):
                outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu()
            all_probs.append(probs)
            all_labels.append(labels)

    return torch.cat(all_probs), torch.cat(all_labels)


def plot_probability_distributions(all_probs, all_labels, eval_name, test_name, viz_dir):
    """
    각 클래스별 probability distribution 히스토그램

    - 파란색: 실제 label=0 (해당 패턴 없음)의 예측 확률 분포
    - 빨간색: 실제 label=1 (해당 패턴 있음)의 예측 확률 분포
    - threshold=0.5 선
    """
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()

    colors_neg = "#2196F3"  # 파란색 (negative)
    colors_pos = "#F44336"  # 빨간색 (positive)

    for i in range(NUM_CLASSES):
        ax = axes[i]
        probs_i = all_probs[:, i].numpy()
        labels_i = all_labels[:, i].numpy()

        # label=0과 label=1 분리
        neg_probs = probs_i[labels_i == 0]
        pos_probs = probs_i[labels_i == 1]

        bins = np.linspace(0, 1, 51)

        # 히스토그램
        ax.hist(neg_probs, bins=bins, alpha=0.6, color=colors_neg,
                label=f"Label=0 (n={len(neg_probs)})", density=True, edgecolor="white", linewidth=0.3)
        ax.hist(pos_probs, bins=bins, alpha=0.6, color=colors_pos,
                label=f"Label=1 (n={len(pos_probs)})", density=True, edgecolor="white", linewidth=0.3)

        # threshold 선
        ax.axvline(x=0.5, color="black", linestyle="--", linewidth=1.5, label="Threshold=0.5")

        # 통계
        neg_mean = neg_probs.mean() if len(neg_probs) > 0 else 0
        pos_mean = pos_probs.mean() if len(pos_probs) > 0 else 0
        separation = pos_mean - neg_mean

        # overlap 계산 (threshold 기준 오분류율)
        fp_rate = (neg_probs >= 0.5).mean() if len(neg_probs) > 0 else 0
        fn_rate = (pos_probs < 0.5).mean() if len(pos_probs) > 0 else 0

        ax.set_title(f"{DEFECT_CLASSES[i]}\n"
                     f"sep={separation:.3f}  FPR={fp_rate:.1%}  FNR={fn_rate:.1%}",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7, loc="upper center")
        ax.set_xlim(-0.02, 1.02)
        ax.grid(alpha=0.2)

    # 마지막 셀: 요약 통계
    ax = axes[NUM_CLASSES]
    ax.axis("off")
    summary_lines = [f"[{eval_name}] {test_name}\n"]
    summary_lines.append(f"{'Class':<14s} {'Sep':>6s} {'FPR':>6s} {'FNR':>6s}")
    summary_lines.append("-" * 36)
    for i in range(NUM_CLASSES):
        probs_i = all_probs[:, i].numpy()
        labels_i = all_labels[:, i].numpy()
        neg_probs = probs_i[labels_i == 0]
        pos_probs = probs_i[labels_i == 1]
        neg_mean = neg_probs.mean() if len(neg_probs) > 0 else 0
        pos_mean = pos_probs.mean() if len(pos_probs) > 0 else 0
        fp_rate = (neg_probs >= 0.5).mean() if len(neg_probs) > 0 else 0
        fn_rate = (pos_probs < 0.5).mean() if len(pos_probs) > 0 else 0
        sep = pos_mean - neg_mean
        summary_lines.append(f"{DEFECT_CLASSES[i]:<14s} {sep:>5.3f} {fp_rate:>5.1%} {fn_rate:>5.1%}")

    ax.text(0.1, 0.95, "\n".join(summary_lines), transform=ax.transAxes,
            fontsize=10, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.suptitle(f"Probability Distribution per Binary Classifier\n{eval_name} - {test_name}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    fname = f"prob_dist_{eval_name.lower().replace(' ', '_')}_{test_name.lower().replace(' ', '_')}.png"
    plt.savefig(viz_dir / fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [저장] {viz_dir / fname}")
    return fname


def plot_combined_overview(all_results_data, viz_dir):
    """4개 평가 x 2개 테스트셋의 separation 종합 비교"""
    fig, axes = plt.subplots(2, 1, figsize=(18, 12))

    eval_names = ["Eval1", "Eval2", "Eval3", "Eval4"]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    for test_idx, test_type in enumerate(["Composite", "Single"]):
        ax = axes[test_idx]
        x = np.arange(NUM_CLASSES)
        width = 0.18

        for eval_idx, ename in enumerate(eval_names):
            key = f"{ename}_{test_type}"
            if key not in all_results_data:
                continue
            seps = all_results_data[key]["separations"]
            fnrs = all_results_data[key]["fnrs"]

            bars = ax.bar(x + eval_idx * width, seps, width,
                         label=ename, color=colors[eval_idx], alpha=0.8)

            # FNR 표시 (작은 텍스트)
            for j, (bar, fnr) in enumerate(zip(bars, fnrs)):
                if fnr > 0.01:  # 1% 이상만 표시
                    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                           f"{fnr:.0%}", ha="center", va="bottom", fontsize=6, color="red")

        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(DEFECT_CLASSES, fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Mean Separation (μ₁ - μ₀)")
        ax.set_title(f"{test_type} Test - Probability Separation per Class\n"
                     f"(빨간 숫자: False Negative Rate)", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(y=0.9, color="green", linestyle=":", alpha=0.5, label="Good separation")

    plt.suptitle("Binary Classifier Discriminability Overview\n"
                 "Mean Separation = E[P(x)|label=1] - E[P(x)|label=0]  (높을수록 좋음)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(viz_dir / "prob_dist_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [저장] {viz_dir / 'prob_dist_overview.png'}")


def main():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    run_dir = Path("runs/evaluation")
    viz_dir = run_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path("data/eval_dataset")
    splits = {
        "test_single": str(data_dir / "test_single"),
        "test_composite": str(data_dir / "test_composite"),
    }

    loader_kwargs = {}
    if use_cuda:
        loader_kwargs = {"pin_memory": True, "num_workers": 4, "persistent_workers": True}

    # 테스트 데이터 로드
    test_single_ds = FBMDataset(splits["test_single"], augment=False)
    test_composite_ds = FBMDataset(splits["test_composite"], augment=False)
    test_single_loader = DataLoader(test_single_ds, batch_size=64, shuffle=False, **loader_kwargs)
    test_composite_loader = DataLoader(test_composite_ds, batch_size=64, shuffle=False, **loader_kwargs)

    print("=" * 60)
    print("  Binary Classifier Probability Distribution 분석")
    print("=" * 60)

    eval_configs = [
        ("Eval1", "eval1_best.pt", FBMClassifier),
        ("Eval2", "eval2_best.pt", FBMClassifier),
        ("Eval3", "eval3_best.pt", FBMClassifier),
        ("Eval4", "eval4_best.pt", FBMDetectionClassifier),
    ]

    all_results_data = {}

    for eval_name, ckpt_name, model_cls in eval_configs:
        ckpt_path = run_dir / ckpt_name
        if not ckpt_path.exists():
            print(f"  [스킵] {ckpt_path} 없음")
            continue

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model = model_cls(num_classes=ckpt["num_classes"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        print(f"\n  [{eval_name}] 모델 로드 완료")

        for test_name, loader in [("Composite", test_composite_loader), ("Single", test_single_loader)]:
            probs, labels = collect_probabilities(model, loader, device, use_cuda)
            plot_probability_distributions(probs, labels, eval_name, test_name, viz_dir)

            # 통계 수집
            seps = []
            fprs = []
            fnrs = []
            for i in range(NUM_CLASSES):
                p = probs[:, i].numpy()
                l = labels[:, i].numpy()
                neg_p = p[l == 0]
                pos_p = p[l == 1]
                neg_mean = neg_p.mean() if len(neg_p) > 0 else 0
                pos_mean = pos_p.mean() if len(pos_p) > 0 else 0
                seps.append(pos_mean - neg_mean)
                fprs.append((neg_p >= 0.5).mean() if len(neg_p) > 0 else 0)
                fnrs.append((pos_p < 0.5).mean() if len(pos_p) > 0 else 0)

            all_results_data[f"{eval_name}_{test_name}"] = {
                "separations": seps, "fprs": fprs, "fnrs": fnrs,
            }

    # 종합 비교 차트
    print(f"\n  종합 비교 차트 생성 중...")
    plot_combined_overview(all_results_data, viz_dir)

    # 결과 출력
    print("\n" + "=" * 80)
    print("  Probability Separation 요약 (Composite Test)")
    print("=" * 80)
    print(f"  {'Class':<14s}", end="")
    for ename in ["Eval1", "Eval2", "Eval3", "Eval4"]:
        print(f"  {ename:>12s}", end="")
    print()
    print(f"  {'─' * 66}")
    for i in range(NUM_CLASSES):
        print(f"  {DEFECT_CLASSES[i]:<14s}", end="")
        for ename in ["Eval1", "Eval2", "Eval3", "Eval4"]:
            key = f"{ename}_Composite"
            if key in all_results_data:
                sep = all_results_data[key]["separations"][i]
                fnr = all_results_data[key]["fnrs"][i]
                print(f"  {sep:>5.3f}({fnr:>4.0%})", end="")
            else:
                print(f"  {'N/A':>12s}", end="")
        print()

    print(f"\n  Sep = Mean Separation (높을수록 구분 good)")
    print(f"  (%) = False Negative Rate (낮을수록 good)")
    print("=" * 80)


if __name__ == "__main__":
    main()
