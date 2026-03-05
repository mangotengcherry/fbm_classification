"""
FBM(Fail Bit Map) 합성 데이터 생성기 - Multi-Label 버전
단일 패턴 + 2개 패턴 중첩 이미지를 생성합니다.

레이블 형식: CSV (filename, row_line, col_line, corner_rect, nail, edge, block)
  - 단일 패턴: [1,0,0,0,0,0] 등
  - 중첩 패턴: [1,0,0,1,0,0] (row_line + nail) 등
  - 정상:      [0,0,0,0,0,0]

사용법:
    python generate_fbm_data.py
    python generate_fbm_data.py --count 300 --output data/fbm_v2
"""

import argparse
import csv
import random
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# FBM 크기
FBM_H = 46
FBM_W = 128

# 불량 패턴 클래스 (normal 제외, 7개)
DEFECT_CLASSES = ["row_line", "col_line", "corner_rect", "nail", "edge", "block", "scatter"]

PATTERN_NAMES_KR = {
    "row_line":    "로우 라인",
    "col_line":    "컬럼 라인",
    "corner_rect": "모서리 사각형",
    "nail":        "손톱/반달",
    "edge":        "가장자리",
    "block":       "블록",
    "scatter":     "랜덤 산포",
}


def add_noise(fbm: np.ndarray, density: float = 0.005) -> np.ndarray:
    noise = np.random.random(fbm.shape) < density
    return np.clip(fbm + noise.astype(np.uint8), 0, 1)


def generate_row_line() -> np.ndarray:
    fbm = np.zeros((FBM_H, FBM_W), dtype=np.uint8)
    num_lines = random.randint(1, 5)
    rows = random.sample(range(FBM_H), num_lines)
    for r in rows:
        start = random.randint(0, FBM_W // 8)
        end = random.randint(FBM_W * 7 // 8, FBM_W)
        thickness = random.randint(1, 2)
        for t in range(thickness):
            if r + t < FBM_H:
                fbm[r + t, start:end] = 1
        if random.random() > 0.3:
            for _ in range(random.randint(1, 5)):
                fbm[r, random.randint(start, end - 1)] = 0
    return fbm


def generate_col_line() -> np.ndarray:
    fbm = np.zeros((FBM_H, FBM_W), dtype=np.uint8)
    num_lines = random.randint(1, 4)
    cols = random.sample(range(FBM_W), num_lines)
    for c in cols:
        start = random.randint(0, FBM_H // 8)
        end = random.randint(FBM_H * 7 // 8, FBM_H)
        thickness = random.randint(1, 2)
        for t in range(thickness):
            if c + t < FBM_W:
                fbm[start:end, c + t] = 1
    return fbm


def generate_corner_rect() -> np.ndarray:
    fbm = np.zeros((FBM_H, FBM_W), dtype=np.uint8)
    corners = random.sample(range(4), random.randint(1, 2))
    for corner in corners:
        rh = random.randint(FBM_H // 5, FBM_H // 2)
        rw = random.randint(FBM_W // 5, FBM_W // 2)
        if corner == 0:   fbm[0:rh, 0:rw] = 1
        elif corner == 1: fbm[0:rh, FBM_W - rw:FBM_W] = 1
        elif corner == 2: fbm[FBM_H - rh:FBM_H, 0:rw] = 1
        else:             fbm[FBM_H - rh:FBM_H, FBM_W - rw:FBM_W] = 1
        edge_noise = np.random.random(fbm.shape) < 0.15
        boundary = np.zeros_like(fbm)
        boundary[1:, :] |= (fbm[1:, :] != fbm[:-1, :]).astype(np.uint8)
        boundary[:, 1:] |= (fbm[:, 1:] != fbm[:, :-1]).astype(np.uint8)
        fbm = np.where(boundary & edge_noise.astype(np.uint8), 1 - fbm, fbm)
    return fbm


def generate_nail() -> np.ndarray:
    fbm = np.zeros((FBM_H, FBM_W), dtype=np.uint8)
    side = random.choice(["top", "bottom", "left", "right"])
    if side == "top":
        cy, cx = -random.randint(5, 15), random.randint(FBM_W // 4, FBM_W * 3 // 4)
        ry, rx = random.randint(FBM_H // 3, FBM_H * 2 // 3), random.randint(FBM_W // 4, FBM_W // 2)
    elif side == "bottom":
        cy, cx = FBM_H + random.randint(5, 15), random.randint(FBM_W // 4, FBM_W * 3 // 4)
        ry, rx = random.randint(FBM_H // 3, FBM_H * 2 // 3), random.randint(FBM_W // 4, FBM_W // 2)
    elif side == "left":
        cy, cx = random.randint(FBM_H // 4, FBM_H * 3 // 4), -random.randint(10, 40)
        ry, rx = random.randint(FBM_H // 4, FBM_H // 2), random.randint(FBM_W // 3, FBM_W * 2 // 3)
    else:
        cy, cx = random.randint(FBM_H // 4, FBM_H * 3 // 4), FBM_W + random.randint(10, 40)
        ry, rx = random.randint(FBM_H // 4, FBM_H // 2), random.randint(FBM_W // 3, FBM_W * 2 // 3)
    for y in range(FBM_H):
        for x in range(FBM_W):
            if ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0:
                fbm[y, x] = 1
    boundary_noise = np.random.random(fbm.shape) < 0.2
    boundary = np.zeros_like(fbm)
    boundary[1:, :] |= (fbm[1:, :] != fbm[:-1, :]).astype(np.uint8)
    boundary[:, 1:] |= (fbm[:, 1:] != fbm[:, :-1]).astype(np.uint8)
    fbm = np.where(boundary & boundary_noise.astype(np.uint8), 1 - fbm, fbm)
    return fbm


def generate_edge() -> np.ndarray:
    fbm = np.zeros((FBM_H, FBM_W), dtype=np.uint8)
    sides = random.sample(["top", "bottom", "left", "right"], random.randint(1, 2))
    for side in sides:
        if side == "top":
            fbm[0:random.randint(2, max(3, FBM_H // 5)), :] = 1
        elif side == "bottom":
            fbm[FBM_H - random.randint(2, max(3, FBM_H // 5)):FBM_H, :] = 1
        elif side == "left":
            fbm[:, 0:random.randint(2, max(3, FBM_W // 8))] = 1
        else:
            fbm[:, FBM_W - random.randint(2, max(3, FBM_W // 8)):FBM_W] = 1
    return fbm


def generate_block() -> np.ndarray:
    fbm = np.zeros((FBM_H, FBM_W), dtype=np.uint8)
    for _ in range(random.randint(1, 2)):
        bh = random.randint(FBM_H // 4, FBM_H * 2 // 3)
        bw = random.randint(FBM_W // 4, FBM_W * 2 // 3)
        y0 = random.randint(0, FBM_H - bh)
        x0 = random.randint(0, FBM_W - bw)
        fbm[y0:y0 + bh, x0:x0 + bw] = 1
    return fbm


def generate_scatter() -> np.ndarray:
    """랜덤 산포 패턴 - 이미지 전체에 무작위로 결함 픽셀이 흩뿌려짐"""
    fbm = np.zeros((FBM_H, FBM_W), dtype=np.uint8)
    # 기본 랜덤 산포 (밀도 5~15%)
    density = random.uniform(0.05, 0.15)
    scatter_mask = np.random.random((FBM_H, FBM_W)) < density
    fbm[scatter_mask] = 1
    # 소규모 클러스터 추가 (현실감)
    num_clusters = random.randint(5, 15)
    for _ in range(num_clusters):
        cy = random.randint(0, FBM_H - 1)
        cx = random.randint(0, FBM_W - 1)
        radius = random.randint(1, 3)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy * dy + dx * dx <= radius * radius:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < FBM_H and 0 <= nx < FBM_W:
                        if random.random() < 0.7:
                            fbm[ny, nx] = 1
    return fbm


# 패턴 생성 함수 매핑
PATTERN_GENERATORS = {
    "row_line":    generate_row_line,
    "col_line":    generate_col_line,
    "corner_rect": generate_corner_rect,
    "nail":        generate_nail,
    "edge":        generate_edge,
    "block":       generate_block,
    "scatter":     generate_scatter,
}


def make_label_vector(pattern_names: list) -> list:
    """패턴 이름 리스트 → [0,1,0,1,...] binary 벡터"""
    return [1 if c in pattern_names else 0 for c in DEFECT_CLASSES]


def generate_composite_fbm(pattern_names: list) -> np.ndarray:
    """여러 패턴을 OR 합성하여 중첩 FBM 생성"""
    fbm = np.zeros((FBM_H, FBM_W), dtype=np.uint8)
    for name in pattern_names:
        layer = PATTERN_GENERATORS[name]()
        fbm = np.clip(fbm + layer, 0, 1)
    fbm = add_noise(fbm, 0.003)
    return fbm


def fbm_to_image(fbm: np.ndarray) -> Image.Image:
    return Image.fromarray((fbm * 255).astype(np.uint8), mode="L")


def save_sample_sheet(output_dir: Path):
    """참고용 샘플 이미지 저장 (단일 + 중첩)"""
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    SCALE = 4
    PADDING = 8
    LABEL_H = 28
    SAMPLES = 5

    print("\n  참고용 샘플 이미지 생성 중...")
    all_strips = []

    # 단일 패턴
    for name, gen_func in PATTERN_GENERATORS.items():
        kr = PATTERN_NAMES_KR[name]
        d = samples_dir / name
        d.mkdir(parents=True, exist_ok=True)
        imgs = []
        for i in range(SAMPLES):
            fbm = gen_func()
            fbm = add_noise(fbm, 0.003)
            img = fbm_to_image(fbm)
            scaled = img.resize((FBM_W * SCALE, FBM_H * SCALE), Image.NEAREST)
            scaled.save(d / f"sample_{i+1}.png")
            imgs.append(scaled)
        strip = _make_strip(f"{name}  ({kr})", imgs, SCALE, PADDING, LABEL_H)
        strip.save(samples_dir / f"{name}_samples.png")
        all_strips.append(strip)
        print(f"    {name:20s} -> {d}")

    # 중첩 패턴 (대표 조합 5개)
    combos = [
        ("row_line", "col_line"),
        ("row_line", "nail"),
        ("corner_rect", "edge"),
        ("nail", "block"),
        ("col_line", "corner_rect"),
    ]
    for c1, c2 in combos:
        d = samples_dir / f"{c1}+{c2}"
        d.mkdir(parents=True, exist_ok=True)
        imgs = []
        for i in range(SAMPLES):
            fbm = generate_composite_fbm([c1, c2])
            img = fbm_to_image(fbm)
            scaled = img.resize((FBM_W * SCALE, FBM_H * SCALE), Image.NEAREST)
            scaled.save(d / f"sample_{i+1}.png")
            imgs.append(scaled)
        label = f"{c1} + {c2}  (중첩)"
        strip = _make_strip(label, imgs, SCALE, PADDING, LABEL_H)
        strip.save(samples_dir / f"{c1}+{c2}_samples.png")
        all_strips.append(strip)
        print(f"    {c1}+{c2:14s} -> {d}")

    # overview
    if all_strips:
        ow = max(s.width for s in all_strips)
        oh = sum(s.height + PADDING for s in all_strips) - PADDING
        overview = Image.new("RGB", (ow, oh), (20, 20, 20))
        y = 0
        for s in all_strips:
            overview.paste(s, (0, y))
            y += s.height + PADDING
        overview.save(samples_dir / "_overview_all_patterns.png")
        print(f"\n  전체 패턴 시트: {samples_dir / '_overview_all_patterns.png'}")
    print(f"  샘플 폴더: {samples_dir}")


def _make_strip(label, images, scale, padding, label_h):
    n = len(images)
    w = n * (FBM_W * scale + padding) - padding
    h = FBM_H * scale + label_h
    strip = Image.new("RGB", (w, h), (30, 30, 30))
    draw = ImageDraw.Draw(strip)
    draw.text((4, 2), label, fill=(100, 200, 255))
    for i, img in enumerate(images):
        strip.paste(img.convert("RGB"), (i * (FBM_W * scale + padding), label_h))
    return strip


def generate_dataset(output_dir: str, count_per_class: int = 300, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    output_path = Path(output_dir)
    train_ratio = 0.8

    # C(6,2) = 15개 조합
    all_combos = list(combinations(DEFECT_CLASSES, 2))
    combo_count = max(30, count_per_class // 5)  # 조합당 생성 수

    single_total = count_per_class * len(DEFECT_CLASSES) + count_per_class  # +normal
    combo_total = combo_count * len(all_combos)
    total_images = single_total + combo_total

    print("=" * 60)
    print("  FBM 합성 데이터셋 생성 (Multi-Label)")
    print("=" * 60)
    print(f"  출력 경로: {output_path}")
    print(f"  FBM 크기: {FBM_W} x {FBM_H}")
    print(f"  불량 클래스: {DEFECT_CLASSES}")
    print(f"  단일 패턴: 클래스당 {count_per_class}장 x {len(DEFECT_CLASSES)}종 + normal {count_per_class}장")
    print(f"  중첩 패턴: 조합당 {combo_count}장 x {len(all_combos)}종")
    print(f"  총 이미지: {total_images}장")
    print(f"  Train/Val: {train_ratio:.0%}/{1 - train_ratio:.0%}")
    print()

    for split in ["train", "val", "test_composite"]:
        (output_path / split / "images").mkdir(parents=True, exist_ok=True)

    train_rows = []
    val_rows = []
    test_rows = []

    # ── 1) 정상 (normal) ──
    print(f"  생성 중: {'normal':20s} ... ", end="", flush=True)
    for i in range(count_per_class):
        fbm = np.zeros((FBM_H, FBM_W), dtype=np.uint8)
        fbm = add_noise(fbm, random.uniform(0.001, 0.01))
        fname = f"normal_{i:04d}.png"
        label = make_label_vector([])

        split = "train" if i < int(count_per_class * train_ratio) else "val"
        fbm_to_image(fbm).save(output_path / split / "images" / fname)
        (train_rows if split == "train" else val_rows).append([fname] + label)
    print(f"{count_per_class}장")

    # ── 2) 단일 패턴 ──
    for pattern_name, gen_func in PATTERN_GENERATORS.items():
        kr = PATTERN_NAMES_KR[pattern_name]
        print(f"  생성 중: {pattern_name:20s} ({kr}) ... ", end="", flush=True)
        label = make_label_vector([pattern_name])

        for i in range(count_per_class):
            fbm = add_noise(gen_func(), 0.003)
            fname = f"single_{pattern_name}_{i:04d}.png"

            split = "train" if i < int(count_per_class * train_ratio) else "val"
            fbm_to_image(fbm).save(output_path / split / "images" / fname)
            (train_rows if split == "train" else val_rows).append([fname] + label)
        print(f"{count_per_class}장")

    # ── 3) 중첩 패턴 (2개 조합) ──
    print()
    for c1, c2 in all_combos:
        combo_name = f"{c1}+{c2}"
        print(f"  생성 중: {combo_name:20s} (중첩) ... ", end="", flush=True)
        label = make_label_vector([c1, c2])

        for i in range(combo_count):
            fbm = generate_composite_fbm([c1, c2])
            fname = f"combo_{c1}+{c2}_{i:04d}.png"

            # train에 70%, val에 15%, test_composite에 15%
            if i < int(combo_count * 0.70):
                split = "train"
                fbm_to_image(fbm).save(output_path / "train" / "images" / fname)
                train_rows.append([fname] + label)
            elif i < int(combo_count * 0.85):
                split = "val"
                fbm_to_image(fbm).save(output_path / "val" / "images" / fname)
                val_rows.append([fname] + label)
            else:
                split = "test_composite"
                fbm_to_image(fbm).save(output_path / "test_composite" / "images" / fname)
                test_rows.append([fname] + label)

        print(f"{combo_count}장")

    # ── CSV 저장 ──
    header = ["filename"] + DEFECT_CLASSES
    for split, rows in [("train", train_rows), ("val", val_rows), ("test_composite", test_rows)]:
        csv_path = output_path / split / "labels.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

    print(f"\n  Train: {len(train_rows)}장  |  Val: {len(val_rows)}장  |  Test(중첩): {len(test_rows)}장")

    # 클래스 목록 저장
    (output_path / "classes.txt").write_text("\n".join(DEFECT_CLASSES), encoding="utf-8")

    # 참고 샘플 이미지
    save_sample_sheet(output_path)

    print()
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="FBM 합성 데이터셋 생성 (Multi-Label)")
    parser.add_argument("--output", type=str, default="data/fbm_dataset", help="출력 디렉토리")
    parser.add_argument("--count", type=int, default=300, help="단일패턴 클래스당 이미지 수")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    args = parser.parse_args()
    generate_dataset(output_dir=args.output, count_per_class=args.count, seed=args.seed)


if __name__ == "__main__":
    main()
