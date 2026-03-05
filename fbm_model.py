"""
FBM 불량 패턴 Multi-Label 분류 CNN 모델

입력: 1 x 46 x 128 (channels x height x width)
출력: num_classes 개의 logit (sigmoid 적용 전)
      각 출력이 독립적으로 해당 패턴 존재 여부를 나타냄
"""

import torch
import torch.nn as nn


# 불량 패턴 클래스 (normal 제외, 7개)
DEFECT_CLASSES = ["row_line", "col_line", "corner_rect", "nail", "edge", "block", "scatter"]


class FBMClassifier(nn.Module):
    """
    FBM 불량 패턴 Multi-Label 분류 CNN

    출력: num_classes개의 logit
      - 학습 시: BCEWithLogitsLoss 사용 (내부에서 sigmoid 적용)
      - 추론 시: sigmoid → threshold 로 다중 패턴 감지
    """

    def __init__(self, num_classes: int = 7):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x  # raw logits


def load_model(model_path: str, device: str = "cpu") -> tuple:
    """저장된 모델과 클래스 목록을 로드합니다."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]

    model = FBMClassifier(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names
