import cv2
import numpy as np
import pytest

from fin_extractor.preprocessing import geometry
from fin_extractor.preprocessing.binarize import binarize


def _fake_document() -> np.ndarray:
    img = np.full((800, 600), 255, np.uint8)
    for i, y in enumerate(range(60, 760, 40)):
        cv2.putText(img, f"INVOICE LINE {i} TOTAL 1,234.00", (40, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
    return img


@pytest.mark.parametrize("angle", [-7.0, -2.5, 0.0, 3.0, 8.0])
def test_estimate_skew_recovers_rotation(angle):
    skewed = geometry.rotate(_fake_document(), -angle)
    assert geometry.estimate_skew(skewed) == pytest.approx(angle, abs=0.3)


@pytest.mark.parametrize("method", ["otsu", "adaptive", "sauvola"])
def test_binarize_outputs_two_levels(method):
    noisy = _fake_document().astype(np.int16) + np.random.default_rng(0).integers(-30, 30, (800, 600))
    out = binarize(np.clip(noisy, 0, 255).astype(np.uint8), method)
    assert set(np.unique(out)) <= {0, 255}


class _FakeOrientationModel:
    """ป้ายที่ Paddle คืน = มุมที่ภาพถูกหมุนตามเข็มอยู่ (วัดกับชุดสังเคราะห์)"""

    def __init__(self, label: str):
        self.label = label

    def predict(self, img, batch_size=1):
        yield {"label_names": [self.label]}


@pytest.mark.parametrize("label, fix", [("0", 0), ("90", 270), ("180", 180), ("270", 90)])
def test_detect_orientation_converts_label_to_fix_angle(monkeypatch, label, fix):
    monkeypatch.setattr(geometry, "_load_orientation_model", lambda: _FakeOrientationModel(label))
    assert geometry.detect_orientation(_fake_document()) == fix


def test_detect_orientation_returns_zero_when_model_fails(monkeypatch):
    def boom():
        raise RuntimeError("no model")

    monkeypatch.setattr(geometry, "_load_orientation_model", boom)
    assert geometry.detect_orientation(_fake_document()) == 0
