"""ขั้น orient: ตรวจทิศทั้งหน้า (0/90/180/270) ด้วยตัวจำแนกทิศของ Paddle แล้วหมุนกลับ."""

from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)

_ROTATE_CODES = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


_orientation_model = None


def _load_orientation_model():
    """โหลดตัวจำแนกทิศโดยไม่ให้ torch ถูกโหลดพ่วง (cuDNN ของ torch กับ Paddle ชนกัน ดู ``paddle_engine.without_modelscope``)."""
    global _orientation_model
    if _orientation_model is None:
        from ..readers.paddle import prepare_paddle_env, without_modelscope

        prepare_paddle_env()
        with without_modelscope():
            from paddlex import create_model

        _orientation_model = create_model(model_name="PP-LCNet_x1_0_doc_ori")
    return _orientation_model


def detect_orientation(img: np.ndarray) -> int:
    """คืนมุม (0/90/180/270) ที่ต้องหมุนตามเข็มนาฬิกาเพื่อให้ภาพตั้งตรง ด้วยตัวจำแนกทิศทั้งหน้าของ Paddle.

    โมเดลคืน "มุมที่ภาพถูกหมุนตามเข็มอยู่" (วัดกับชุดสังเคราะห์: ป้าย 90 = ภาพถูกหมุนตามเข็ม 90°)
    มุมแก้จึงเป็น 360 − ป้าย ถ้าโหลดโมเดลไม่ได้จะคืน 0 เพื่อไม่ให้ pipeline พัง (ผู้เรียกที่ต้องรู้ว่าล้มใช้ ``detect_orientation_checked``)
    """
    return detect_orientation_checked(img)[0]


def detect_orientation_checked(img: np.ndarray) -> tuple[int, str | None]:
    """(มุมที่ต้องหมุน, ข้อผิดพลาด) ข้อผิดพลาด = ``None`` เมื่อตรวจได้ ถ้าโมเดลล้ม (เช่น Paddle โหลด cuDNN ไม่ได้) คืน (0, ข้อความ) ไม่ใช่ 0 เงียบ ๆ

    มุม 0 จากการตรวจที่ล้มต่างจากมุม 0 จากภาพที่ตั้งตรงจริง: ภาพตะแคงจะผ่านไปทุกขั้นถัดไปโดยไม่มีสัญญาณ (เจอจริงกับ 1315)
    """
    try:
        model = _load_orientation_model()
        rgb = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        label = int(next(iter(model.predict(rgb, batch_size=1)))["label_names"][0])
    except Exception as e:  # noqa: BLE001 — โมเดลล้มได้หลายแบบ ไม่ควรทำให้ pipeline พัง
        log.warning("orientation detection skipped: %s", e)
        return 0, f"{type(e).__name__}: {str(e)[:200]}"
    return (360 - label) % 360, None


def rotate_orthogonal(img: np.ndarray, angle: int) -> np.ndarray:
    return img if angle == 0 else cv2.rotate(img, _ROTATE_CODES[angle])
