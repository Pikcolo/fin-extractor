"""แก้การหมุน/เอียงของภาพ และตัดขอบ."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .enhance import to_gray

log = logging.getLogger(__name__)

_ROTATE_CODES = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


_orientation_model = None


def _load_orientation_model():
    global _orientation_model
    if _orientation_model is None:
        from ..ocr.paddle_engine import prepare_paddle_env

        prepare_paddle_env()
        from paddlex import create_model

        _orientation_model = create_model(model_name="PP-LCNet_x1_0_doc_ori")
    return _orientation_model


def detect_orientation(img: np.ndarray) -> int:
    """คืนมุม (0/90/180/270) ที่ต้องหมุนตามเข็มนาฬิกาเพื่อให้ภาพตั้งตรง ด้วยตัวจำแนกทิศทั้งหน้าของ Paddle.

    โมเดลคืน "มุมที่ภาพถูกหมุนตามเข็มอยู่" (วัดกับชุดสังเคราะห์: ป้าย 90 = ภาพถูกหมุนตามเข็ม 90°)
    มุมแก้จึงเป็น 360 − ป้าย ถ้าโหลดโมเดลไม่ได้จะคืน 0 เพื่อไม่ให้ pipeline พัง
    """
    try:
        model = _load_orientation_model()
        rgb = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        label = int(next(iter(model.predict(rgb, batch_size=1)))["label_names"][0])
    except Exception as e:  # noqa: BLE001 — โมเดลล้มได้หลายแบบ ไม่ควรทำให้ pipeline พัง
        log.warning("orientation detection skipped: %s", e)
        return 0
    return (360 - label) % 360


def rotate_orthogonal(img: np.ndarray, angle: int) -> np.ndarray:
    return img if angle == 0 else cv2.rotate(img, _ROTATE_CODES[angle])


def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    """หมุนภาพทวนเข็มนาฬิกา ``angle`` องศา โดยขยาย canvas ไม่ให้มุมโดนตัด และเติมพื้นขาว."""
    if abs(angle) < 1e-3:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += new_w / 2 - w / 2
    m[1, 2] += new_h / 2 - h / 2
    border = (255,) * (3 if img.ndim == 3 else 1)
    return cv2.warpAffine(
        img, m, (new_w, new_h), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=border,
    )


def _projection_score(binary: np.ndarray, angle: float) -> float:
    # บรรทัดข้อความที่ตรง → ปริมาณหมึกต่อแถวมีพีคคม → ผลต่างระหว่างแถวติดกันสูง
    # rotate เติมขอบสีขาว จึงต้องกลับสีหลังหมุน ไม่ให้ขอบถูกนับเป็นหมึก
    ink = 255 - rotate(binary, angle)
    rows = ink.sum(axis=1, dtype=np.float64)
    return float(np.sum(np.diff(rows) ** 2))


def estimate_skew(img: np.ndarray, max_angle: float = 15.0) -> float:
    """ประมาณมุมเอียงด้วย projection profile (coarse 1° → fine 0.1°).

    ใช้ได้ทั้งเอกสารมีตารางและไม่มีตาราง; คืนมุมที่ต้องหมุน (ทวนเข็มนาฬิกา) เพื่อแก้เอียง
    """
    gray = to_gray(img)
    scale = 1000 / max(gray.shape)
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    def score(a: float) -> float:
        return _projection_score(binary, a)

    coarse = np.arange(-max_angle, max_angle + 0.5, 1.0)
    best = max(coarse, key=score)
    fine = np.arange(best - 1.0, best + 1.0 + 1e-9, 0.1)
    return round(float(max(fine, key=score)), 2)


def deskew(img: np.ndarray, max_angle: float = 15.0, min_angle: float = 0.1) -> tuple[np.ndarray, float]:
    angle = estimate_skew(img, max_angle)
    if abs(angle) < min_angle:
        return img, 0.0
    return rotate(img, angle), angle


def crop_to_content(img: np.ndarray, margin: int = 20) -> np.ndarray:
    """ตัดขอบว่าง/ขอบดำจากการสแกนออก โดยหา bounding box ของหมึกทั้งหมด."""
    gray = to_gray(img)
    _, inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # ขอบดำของสแกนเนอร์จะติดขอบภาพ — ลบ component ที่แตะขอบก่อน
    h, w = inv.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    for i in range(1, n):
        x, y, bw, bh, _ = stats[i]
        if x == 0 or y == 0 or x + bw >= w or y + bh >= h:
            inv[labels == i] = 0
    pts = cv2.findNonZero(inv)
    if pts is None:
        return img
    x, y, bw, bh = cv2.boundingRect(pts)
    x0, y0 = max(x - margin, 0), max(y - margin, 0)
    x1, y1 = min(x + bw + margin, w), min(y + bh + margin, h)
    return img[y0:y1, x0:x1]
