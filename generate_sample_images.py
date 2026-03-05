"""README용 샘플 패턴 이미지 생성"""
import random
import numpy as np
from PIL import Image
from generate_fbm_data import (
    PATTERN_GENERATORS, PATTERN_NAMES_KR, FBM_H, FBM_W,
    add_noise, generate_composite_fbm, fbm_to_image,
)

random.seed(42)
np.random.seed(42)

SCALE = 4
out = "docs/images"

from pathlib import Path
Path(out).mkdir(parents=True, exist_ok=True)

# 단일 패턴 샘플 3장씩
for name, gen in PATTERN_GENERATORS.items():
    imgs = []
    for i in range(3):
        fbm = add_noise(gen(), 0.003)
        img = fbm_to_image(fbm).resize((FBM_W * SCALE, FBM_H * SCALE), Image.NEAREST)
        imgs.append(img)
    # 가로 연결
    w = FBM_W * SCALE
    h = FBM_H * SCALE
    strip = Image.new("L", (w * 3 + 8, h), 0)
    for i, img in enumerate(imgs):
        strip.paste(img, (i * (w + 4), 0))
    strip.save(f"{out}/sample_{name}.png")

# 중첩 패턴 샘플
combos = [("row_line", "nail"), ("corner_rect", "block"), ("col_line", "scatter"), ("edge", "scatter")]
for c1, c2 in combos:
    imgs = []
    for i in range(3):
        fbm = generate_composite_fbm([c1, c2])
        img = fbm_to_image(fbm).resize((FBM_W * SCALE, FBM_H * SCALE), Image.NEAREST)
        imgs.append(img)
    w = FBM_W * SCALE
    h = FBM_H * SCALE
    strip = Image.new("L", (w * 3 + 8, h), 0)
    for i, img in enumerate(imgs):
        strip.paste(img, (i * (w + 4), 0))
    strip.save(f"{out}/sample_{c1}+{c2}.png")

print("샘플 이미지 생성 완료")
