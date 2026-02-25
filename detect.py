"""
FBM 불량 패턴 Multi-Label 분류 - CLI 추론 스크립트

사용법:
    python detect.py --source path/to/fbm_image.png
    python detect.py --source path/to/folder/
    python detect.py --threshold 0.3
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from fbm_model import FBMClassifier, DEFECT_CLASSES

PATTERN_NAMES_KR = {
    "row_line": "로우 라인", "col_line": "컬럼 라인", "corner_rect": "모서리 사각형",
    "nail": "손톱/반달", "edge": "가장자리", "block": "블록",
}


def predict_multilabel(model, img_path: str, class_names: list, device, threshold=0.5):
    """Multi-label 예측: sigmoid → threshold"""
    img = Image.open(img_path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0].cpu()

    detected = []
    all_probs = {}
    for i, name in enumerate(class_names):
        p = probs[i].item()
        all_probs[name] = p
        if p >= threshold:
            detected.append((name, p))

    detected.sort(key=lambda x: -x[1])
    return detected, all_probs


def detect(model_path: str, source: str, threshold: float):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  FBM 불량 패턴 Multi-Label 분류 - 추론")
    print("=" * 60)

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]

    model = FBMClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"  모델: {model_path}")
    print(f"  임계값: {threshold}")
    print()

    source_path = Path(source)
    if source_path.is_dir():
        image_paths = sorted(
            p for p in source_path.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        )
    elif source_path.is_file():
        image_paths = [source_path]
    else:
        print(f"  [오류] 경로를 찾을 수 없습니다: {source}")
        return

    print(f"  이미지: {len(image_paths)}장")
    print("─" * 60)

    for img_path in image_paths:
        detected, all_probs = predict_multilabel(model, str(img_path), class_names, device, threshold)

        print(f"\n  {img_path.name}")
        if not detected:
            print(f"     → 정상 (불량 패턴 미감지)")
        else:
            labels = ", ".join(f"{n}({p:.0%})" for n, p in detected)
            print(f"     → 감지: {labels}")
            if len(detected) >= 2:
                print(f"     [!] 중첩 불량 ({len(detected)}개 패턴)")

        for name in class_names:
            p = all_probs[name]
            kr = PATTERN_NAMES_KR.get(name, name)
            mark = " <<" if p >= threshold else ""
            bar = "#" * int(p * 20)
            print(f"     {name:14s} {p:5.1%}  {bar}{mark}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="FBM Multi-Label 추론")
    parser.add_argument("--model", type=str, default="runs/fbm_train/best.pt")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    detect(model_path=args.model, source=args.source, threshold=args.threshold)


if __name__ == "__main__":
    main()
