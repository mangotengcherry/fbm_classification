"""
FBM 불량 패턴 Multi-Label 분류 - GUI 애플리케이션

사용법:
    python webcam_detect.py
    python webcam_detect.py --model runs/fbm_train/best.pt
"""

import argparse
import tkinter as tk
from tkinter import filedialog, ttk
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageTk

from fbm_model import FBMClassifier

PATTERN_NAMES_KR = {
    "row_line": "로우 라인", "col_line": "컬럼 라인", "corner_rect": "모서리 사각형",
    "nail": "손톱/반달", "edge": "가장자리", "block": "블록",
}

PATTERN_COLORS = {
    "row_line": "#F44336", "col_line": "#FF9800", "corner_rect": "#E91E63",
    "nail": "#9C27B0", "edge": "#2196F3", "block": "#FF5722",
}


class FBMDetectorApp:
    def __init__(self, model_path: str = "runs/fbm_train/best.pt"):
        self.root = tk.Tk()
        self.root.title("FBM Multi-Label 불량 패턴 분류기")
        self.root.configure(bg="#1e1e1e")

        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = 0.5

        self.model = None
        self.class_names = []
        self._load_model()

        self.current_image_path = None
        self.original_pil = None

        self._build_ui()
        self._center_window(950, 700)

    def _load_model(self):
        if not Path(self.model_path).exists():
            self.model = None
            self.class_names = list(PATTERN_NAMES_KR.keys())
            return
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        self.class_names = checkpoint["class_names"]
        self.model = FBMClassifier(num_classes=checkpoint["num_classes"]).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def _center_window(self, w, h):
        x = (self.root.winfo_screenwidth() - w) // 2
        y = (self.root.winfo_screenheight() - h) // 2
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    def _build_ui(self):
        # 상단
        ctrl = tk.Frame(self.root, bg="#2d2d2d", pady=8, padx=10)
        ctrl.pack(fill=tk.X)

        ttk.Button(ctrl, text="이미지 열기", command=self._open_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="폴더 열기", command=self._open_folder).pack(side=tk.LEFT, padx=4)

        tk.Label(ctrl, text="  임계값:", bg="#2d2d2d", fg="white",
                 font=("맑은 고딕", 10)).pack(side=tk.LEFT, padx=(20, 4))

        self.threshold_var = tk.DoubleVar(value=0.5)
        self.threshold_slider = tk.Scale(
            ctrl, from_=0.1, to=0.9, resolution=0.05, orient=tk.HORIZONTAL,
            variable=self.threshold_var, command=self._on_threshold_change, length=160,
            bg="#2d2d2d", fg="white", highlightthickness=0,
            troughcolor="#555555", activebackground="#4fc3f7",
        )
        self.threshold_slider.pack(side=tk.LEFT, padx=4)

        self.thr_label = tk.Label(ctrl, text="0.50", bg="#2d2d2d", fg="#4fc3f7",
                                  font=("맑은 고딕", 11, "bold"), width=4)
        self.thr_label.pack(side=tk.LEFT)

        status_txt = "로드됨" if self.model else "학습 필요"
        tk.Label(ctrl, text=f"모델: {Path(self.model_path).stem} ({status_txt})",
                 bg="#2d2d2d", fg="#888888", font=("맑은 고딕", 9)).pack(side=tk.RIGHT)

        # 메인
        main = tk.Frame(self.root, bg="#1e1e1e")
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 캔버스
        left = tk.Frame(main, bg="#1e1e1e")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(left, text="FBM 이미지", bg="#1e1e1e", fg="#4fc3f7",
                 font=("맑은 고딕", 11, "bold")).pack(anchor=tk.W, pady=(0, 4))
        self.canvas = tk.Canvas(left, bg="#2d2d2d", highlightthickness=1,
                                highlightbackground="#444444")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self._show_placeholder()

        # 결과 패널
        right = tk.Frame(main, bg="#252525", width=300)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        right.pack_propagate(False)

        tk.Label(right, text="감지된 패턴", bg="#252525", fg="#4fc3f7",
                 font=("맑은 고딕", 12, "bold")).pack(anchor=tk.W, padx=12, pady=(12, 4))
        tk.Frame(right, bg="#4fc3f7", height=2).pack(fill=tk.X, padx=12, pady=(0, 10))

        # 감지 결과 요약
        self.detected_label = tk.Label(right, text="—", bg="#252525", fg="white",
                                       font=("맑은 고딕", 14, "bold"), wraplength=260,
                                       justify=tk.LEFT)
        self.detected_label.pack(anchor=tk.W, padx=12, pady=(5, 2))

        self.detected_sub = tk.Label(right, text="", bg="#252525", fg="#aaaaaa",
                                     font=("맑은 고딕", 10), wraplength=260, justify=tk.LEFT)
        self.detected_sub.pack(anchor=tk.W, padx=12, pady=(0, 15))

        # 개별 클래스 확률 바
        tk.Label(right, text="클래스별 확률 (sigmoid)", bg="#252525", fg="#888888",
                 font=("맑은 고딕", 9)).pack(anchor=tk.W, padx=12, pady=(0, 4))

        self.probs_frame = tk.Frame(right, bg="#252525")
        self.probs_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 10))

        # 배치 결과
        self.batch_text = tk.Text(right, bg="#252525", fg="#cccccc", font=("Consolas", 9),
                                  relief=tk.FLAT, wrap=tk.WORD, borderwidth=0, height=8)
        self.batch_text.pack(fill=tk.X, padx=12, pady=(0, 10))
        self.batch_text.pack_forget()

        # 상태바
        self.status_var = tk.StringVar(
            value="이미지를 열어 FBM 불량 패턴을 분류하세요" if self.model
            else "모델이 없습니다. train.py를 먼저 실행하세요"
        )
        tk.Label(self.root, textvariable=self.status_var, bg="#007acc", fg="white",
                 font=("맑은 고딕", 9), anchor=tk.W, padx=10, pady=3).pack(fill=tk.X, side=tk.BOTTOM)

    def _show_placeholder(self):
        self.canvas.delete("all")
        self.canvas.update_idletasks()
        cx = self.canvas.winfo_width() // 2
        cy = self.canvas.winfo_height() // 2
        self.canvas.create_text(cx, cy - 10, text="FBM 이미지를 열어주세요",
                                fill="#666666", font=("맑은 고딕", 14))
        self.canvas.create_text(cx, cy + 15, text="중첩 불량 패턴도 감지합니다",
                                fill="#555555", font=("맑은 고딕", 9))

    def _open_image(self):
        path = filedialog.askopenfilename(
            title="FBM 이미지", filetypes=[("이미지", "*.png *.jpg *.jpeg *.bmp *.tiff")])
        if path:
            self._classify_single(path)

    def _open_folder(self):
        folder = filedialog.askdirectory(title="FBM 이미지 폴더")
        if folder:
            self._classify_batch(folder)

    def _classify_single(self, path: str):
        if not self.model:
            self.status_var.set("모델이 없습니다.")
            return

        self.current_image_path = path
        self.original_pil = Image.open(path).convert("L")
        self._display_image(self.original_pil)

        arr = np.array(self.original_pil, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits)[0].cpu()

        thr = self.threshold_var.get()
        detected = [(self.class_names[i], probs[i].item())
                    for i in range(len(self.class_names)) if probs[i].item() >= thr]
        detected.sort(key=lambda x: -x[1])

        # 결과 표시
        if not detected:
            self.detected_label.config(text="정상", fg="#4CAF50")
            self.detected_sub.config(text="불량 패턴 미감지")
        elif len(detected) == 1:
            name, conf = detected[0]
            kr = PATTERN_NAMES_KR.get(name, name)
            color = PATTERN_COLORS.get(name, "#ffffff")
            self.detected_label.config(text=f"{name} ({conf:.0%})", fg=color)
            self.detected_sub.config(text=kr)
        else:
            names = [f"{n}({p:.0%})" for n, p in detected]
            self.detected_label.config(text=f"중첩 불량 ({len(detected)}개)", fg="#FFD600")
            self.detected_sub.config(text=" + ".join(names))

        # 확률 바
        for w in self.probs_frame.winfo_children():
            w.destroy()

        for i, name in enumerate(self.class_names):
            p = probs[i].item()
            row = tk.Frame(self.probs_frame, bg="#252525")
            row.pack(fill=tk.X, pady=2)

            c = PATTERN_COLORS.get(name, "#888888")
            is_detected = p >= thr
            fg_color = c if is_detected else "#666666"

            tk.Label(row, text=name, bg="#252525", fg=fg_color, width=14, anchor=tk.W,
                     font=("Consolas", 9, "bold" if is_detected else "normal")).pack(side=tk.LEFT)

            bar_canvas = tk.Canvas(row, bg="#333333", height=14, highlightthickness=0, width=100)
            bar_canvas.pack(side=tk.LEFT, padx=4)
            bar_canvas.create_rectangle(0, 0, int(100 * p), 14, fill=c if is_detected else "#555555", outline="")

            # 임계값 선
            thr_x = int(100 * thr)
            bar_canvas.create_line(thr_x, 0, thr_x, 14, fill="#FFD600", width=1)

            mark = " ◀" if is_detected else ""
            tk.Label(row, text=f"{p:.0%}{mark}", bg="#252525", fg=fg_color,
                     font=("Consolas", 9), width=7).pack(side=tk.LEFT)

        self.batch_text.pack_forget()

        fname = Path(path).name
        w, h = self.original_pil.size
        n_det = len(detected)
        det_str = "정상" if n_det == 0 else f"{n_det}개 패턴 감지"
        self.status_var.set(f"{fname} ({w}x{h}) → {det_str}")

    def _classify_batch(self, folder: str):
        if not self.model:
            return

        folder_path = Path(folder)
        paths = sorted(p for p in folder_path.iterdir()
                       if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
        if not paths:
            self.status_var.set(f"이미지 없음: {folder}")
            return

        self.batch_text.pack(fill=tk.X, padx=12, pady=(0, 10))
        self.batch_text.config(state=tk.NORMAL)
        self.batch_text.delete("1.0", tk.END)
        self.batch_text.insert(tk.END, f"배치: {len(paths)}장\n{'─'*35}\n")

        thr = self.threshold_var.get()
        stats = {"정상": 0, "단일": 0, "중첩": 0}

        for p in paths:
            img = Image.open(str(p)).convert("L")
            arr = np.array(img, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(self.device)

            with torch.no_grad():
                probs = torch.sigmoid(self.model(tensor))[0].cpu()

            detected = [self.class_names[i] for i in range(len(self.class_names))
                        if probs[i].item() >= thr]

            if len(detected) == 0:
                stats["정상"] += 1
                label = "정상"
            elif len(detected) == 1:
                stats["단일"] += 1
                label = detected[0]
            else:
                stats["중첩"] += 1
                label = "+".join(detected)

            self.batch_text.insert(tk.END, f"{p.name:30s} → {label}\n")

        self.batch_text.insert(tk.END, f"\n요약: 정상:{stats['정상']} 단일:{stats['단일']} 중첩:{stats['중첩']}\n")
        self.batch_text.config(state=tk.DISABLED)

        self._classify_single(str(paths[0]))
        self.batch_text.pack(fill=tk.X, padx=12, pady=(0, 10))
        self.status_var.set(f"배치 완료: {len(paths)}장")

    def _display_image(self, pil_img):
        self.canvas.update_idletasks()
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        iw, ih = pil_img.size
        scale = min(cw / iw, ch / ih) * 0.9
        scale = max(scale, 1.0)
        display = pil_img.resize((int(iw * scale), int(ih * scale)), Image.NEAREST).convert("RGB")
        self._tk_img = ImageTk.PhotoImage(display)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=self._tk_img)
        self.canvas.create_text(10, 10, text=f"원본: {iw}x{ih}  (x{scale:.0f})",
                                fill="#888888", font=("맑은 고딕", 8), anchor=tk.NW)

    def _on_threshold_change(self, _val):
        self.thr_label.config(text=f"{self.threshold_var.get():.2f}")
        if self.current_image_path:
            self._classify_single(self.current_image_path)

    def _on_canvas_resize(self, _event):
        if self.original_pil:
            self._display_image(self.original_pil)
        else:
            self._show_placeholder()

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="FBM Multi-Label GUI")
    parser.add_argument("--model", type=str, default="runs/fbm_train/best.pt")
    args = parser.parse_args()
    FBMDetectorApp(model_path=args.model).run()


if __name__ == "__main__":
    main()
