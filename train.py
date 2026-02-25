"""
FBM 불량 패턴 Multi-Label 분류 - 학습 스크립트

학습 후 단일 패턴 + 중첩 패턴 테스트셋 평가까지 자동 수행합니다.

사용법:
    python generate_fbm_data.py      # 먼저 데이터 생성
    python train.py                  # 학습 + 평가
    python train.py --epochs 50
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from fbm_model import FBMClassifier, DEFECT_CLASSES

PATTERN_NAMES_KR = {
    "row_line": "로우라인", "col_line": "컬럼라인", "corner_rect": "모서리사각",
    "nail": "손톱/반달", "edge": "가장자리", "block": "블록",
}


class FBMMultiLabelDataset(Dataset):
    """
    Multi-Label FBM 데이터셋 (CSV 레이블)

    디렉토리 구조:
        data_dir/
        ├── images/
        │   ├── single_row_line_0000.png
        │   ├── combo_row_line+nail_0000.png
        │   └── ...
        └── labels.csv
    """

    def __init__(self, data_dir: str, augment: bool = False):
        self.data_dir = Path(data_dir)
        self.augment = augment
        self.samples = []       # (image_path, label_vector)
        self.class_names = []

        csv_path = self.data_dir / "labels.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"labels.csv 없음: {csv_path}")

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            self.class_names = header[1:]  # ['row_line', 'col_line', ...]

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


def print_gpu_info():
    if not torch.cuda.is_available():
        print("  GPU: 사용 불가 (CPU 모드)")
        return False
    props = torch.cuda.get_device_properties(0)
    gpu_mem = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / (1024 ** 3)
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {gpu_mem:.1f} GB  |  CUDA: {torch.version.cuda}")
    return True


def evaluate(model, loader, criterion, device, use_cuda, threshold=0.5):
    """Multi-label 평가: per-label accuracy, exact match, loss"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_cuda):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) >= threshold).float()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    n = all_labels.size(0)

    avg_loss = total_loss / n

    # Per-label accuracy
    per_label_acc = (all_preds == all_labels).float().mean(dim=0)

    # Exact match (모든 레이블이 정확히 일치)
    exact_match = (all_preds == all_labels).all(dim=1).float().mean().item()

    # Sample-level accuracy (개별 레이블 전체 평균)
    sample_acc = (all_preds == all_labels).float().mean().item()

    return avg_loss, sample_acc, exact_match, per_label_acc


def evaluate_composite_test(model, data_dir, device, use_cuda, threshold=0.5):
    """중첩 패턴 테스트셋 상세 평가"""
    test_dir = Path(data_dir) / "test_composite"
    if not test_dir.exists() or not (test_dir / "labels.csv").exists():
        print("  [정보] 중첩 테스트셋이 없습니다.")
        return

    test_dataset = FBMMultiLabelDataset(test_dir, augment=False)
    if len(test_dataset) == 0:
        return

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    criterion = nn.BCEWithLogitsLoss()

    _, sample_acc, exact_match, per_label_acc = evaluate(
        model, test_loader, criterion, device, use_cuda, threshold
    )

    class_names = test_dataset.class_names

    print()
    print("=" * 60)
    print("  중첩 패턴 테스트 결과 (test_composite)")
    print("=" * 60)
    print(f"  테스트 이미지: {len(test_dataset)}장 (모두 2개 패턴 중첩)")
    print(f"  Exact Match (전체 벡터 일치): {exact_match:.1%}")
    print(f"  Label Accuracy (개별 레이블):  {sample_acc:.1%}")
    print()
    print(f"  {'패턴':<20s}  정확도")
    print(f"  {'─' * 35}")
    for i, name in enumerate(class_names):
        kr = PATTERN_NAMES_KR.get(name, name)
        acc = per_label_acc[i].item()
        bar = "#" * int(acc * 20)
        print(f"  {name:<14s} ({kr})  {acc:.1%}  {bar}")
    print("=" * 60)


def train(
    data_dir: str = "data/fbm_dataset",
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 0.001,
    output_dir: str = "runs/fbm_train",
):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print("=" * 60)
    print("  FBM 불량 패턴 Multi-Label 분류 - 학습")
    print("=" * 60)
    print(f"  데이터: {data_dir}")
    has_gpu = print_gpu_info()
    print(f"  에포크: {epochs} | 배치: {batch_size} | LR: {lr}")
    if has_gpu:
        print(f"  Mixed Precision (FP16): 활성화")

    # 데이터 로드
    train_dir = Path(data_dir) / "train"
    val_dir = Path(data_dir) / "val"

    if not (train_dir / "labels.csv").exists():
        print(f"\n  [오류] labels.csv를 찾을 수 없습니다: {train_dir}")
        print("  먼저 실행: python generate_fbm_data.py")
        return

    train_dataset = FBMMultiLabelDataset(train_dir, augment=True)
    val_dataset = FBMMultiLabelDataset(val_dir, augment=False)

    num_classes = len(train_dataset.class_names)
    class_names = train_dataset.class_names

    # 데이터 통계
    train_labels = torch.tensor([s[1] for s in train_dataset.samples])
    n_single = (train_labels.sum(dim=1) == 1).sum().item()
    n_multi = (train_labels.sum(dim=1) >= 2).sum().item()
    n_normal = (train_labels.sum(dim=1) == 0).sum().item()

    print(f"  클래스 ({num_classes}개): {class_names}")
    print(f"  Train: {len(train_dataset)}장 (단일:{n_single} 중첩:{n_multi} 정상:{n_normal})")
    print(f"  Val: {len(val_dataset)}장")
    print(f"  Loss: BCEWithLogitsLoss (multi-label)")
    print()

    loader_kwargs = {}
    if use_cuda:
        loader_kwargs = {"pin_memory": True, "num_workers": 4, "persistent_workers": True}
        torch.backends.cudnn.benchmark = True

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

    # 모델
    model = FBMClassifier(num_classes=num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  모델 파라미터: {total_params:,}개")
    print("─" * 60)

    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    best_val_exact = 0.0
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    history = {"train_loss": [], "val_loss": [], "val_label_acc": [], "val_exact_match": []}
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── Train ──
        model.train()
        train_loss = 0.0
        train_total = 0

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
            train_total += inputs.size(0)

        scheduler.step()
        train_loss /= train_total

        # ── Validation ──
        val_loss, val_acc, val_exact, val_per_label = evaluate(
            model, val_loader, criterion, device, use_cuda
        )

        et = time.time() - t0
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_label_acc"].append(val_acc)
        history["val_exact_match"].append(val_exact)

        gpu_str = ""
        if use_cuda:
            gpu_str = f"  | GPU:{torch.cuda.max_memory_allocated() / 1024**2:.0f}MB"

        marker = ""
        if val_exact > best_val_exact:
            best_val_exact = val_exact
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "num_classes": num_classes,
                "epoch": epoch,
                "val_exact_match": val_exact,
            }, out_path / "best.pt")
            marker = " *"

        print(
            f"  Epoch {epoch:3d}/{epochs}"
            f"  |  Train Loss: {train_loss:.4f}"
            f"  |  Val Loss: {val_loss:.4f}  LabelAcc: {val_acc:.1%}  ExactMatch: {val_exact:.1%}"
            f"  |  {et:.1f}s{gpu_str}{marker}"
        )

    # 최종 저장
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "num_classes": num_classes,
        "epoch": epochs,
        "val_exact_match": val_exact,
    }, out_path / "last.pt")

    total_time = time.time() - start_time
    with open(out_path / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("─" * 60)
    print(f"  학습 완료! ({total_time:.1f}초)")
    print(f"  최고 Val Exact Match: {best_val_exact:.1%}")
    if use_cuda:
        print(f"  디바이스: {torch.cuda.get_device_name(0)}")
    print(f"  모델: {out_path / 'best.pt'}")

    # ── 중첩 테스트셋 평가 ──
    best_ckpt = torch.load(out_path / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(best_ckpt["model_state_dict"])
    evaluate_composite_test(model, data_dir, device, use_cuda)


def main():
    parser = argparse.ArgumentParser(description="FBM Multi-Label 분류 학습")
    parser.add_argument("--data", type=str, default="data/fbm_dataset")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output", type=str, default="runs/fbm_train")
    args = parser.parse_args()

    train(data_dir=args.data, epochs=args.epochs, batch_size=args.batch_size,
          lr=args.lr, output_dir=args.output)


if __name__ == "__main__":
    main()
