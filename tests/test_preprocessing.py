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


def test_detect_orientation_checked_reports_the_failure_instead_of_a_silent_zero(monkeypatch):
    def boom():
        raise OSError("cudnn_cnn64_9.dll")

    monkeypatch.setattr(geometry, "_load_orientation_model", boom)
    angle, err = geometry.detect_orientation_checked(_fake_document())
    assert angle == 0 and "OSError" in err and "cudnn" in err
    monkeypatch.setattr(geometry, "_load_orientation_model", lambda: _FakeOrientationModel("0"))
    assert geometry.detect_orientation_checked(_fake_document()) == (0, None)  # 0 ที่ตรวจได้จริงไม่มีข้อผิดพลาด


def test_preprocess_records_orientation_error_in_meta(monkeypatch):
    from fin_extractor.preprocessing import preprocess

    monkeypatch.setattr(geometry, "_load_orientation_model", lambda: (_ for _ in ()).throw(RuntimeError("no model")))
    res = preprocess(_fake_document(), {"fix_orientation": True, "keep_color": True, "min_long_side": 0})
    assert res.meta["orientation"] == 0 and "RuntimeError" in res.meta["orientation_error"]


def test_preprocess_manual_orientation_rotates_without_loading_the_model(monkeypatch):
    from fin_extractor.preprocessing import preprocess

    monkeypatch.setattr(geometry, "_load_orientation_model", lambda: (_ for _ in ()).throw(AssertionError("ต้องไม่โหลดโมเดล")))
    img = _fake_document()
    res = preprocess(img, {"fix_orientation": True, "keep_color": True, "min_long_side": 0}, orientation=90)
    assert res.image.shape[:2] == img.shape[:2][::-1]  # หมุน 90° สลับกว้าง/สูง
    assert res.meta["orientation"] == 90 and res.meta["orientation_source"] == "manual" and "orientation_error" not in res.meta
    with pytest.raises(ValueError):
        preprocess(img, {"keep_color": True, "min_long_side": 0}, orientation=45)
