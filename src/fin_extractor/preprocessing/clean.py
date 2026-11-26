"""ขั้น clean: ลบสิ่งรบกวนจากการสแกน — เส้นสกปรกพาดขอบถึงขอบ (ก่อนหมุน) และ denoise."""

from __future__ import annotations

import cv2
import numpy as np

from .enhance import to_gray

# เส้นสกปรกจากเครื่องสแกน (ฝุ่นบนแนวเซนเซอร์): แถบเทาจางพาดขอบซ้ายถึงขอบขวาของภาพดิบ
# ค่ามาจาก learning/preprocess_finance (src/finance_ocr/preprocess/crop.py:scanner_streaks, knowledge/preprocess/03_page_geometry.md)
# ซึ่งวัดบนภาพสแกนชุดเดียวกัน (ไฟล์ตรงกันทั้ง 18 ไฟล์) ที่ความละเอียดเดิม 200 dpi (กว้าง 1632 px)
# kernel จึงเก็บเป็นสัดส่วนของความกว้างหน้า: 41×7 px ที่ 1632 px คงไว้เป็น 41/1632 × 7/1632
# ตรวจซ้ำบนภาพ render 300 dpi (กว้าง 2449 px): ขนาด 41×7 ตายตัวเจอแค่ 1322 ส่วนแบบสัดส่วน (62×11) เจอ 1320 และ 1322
# ตำแหน่งเดียวกับที่ 200 dpi และไม่เจอในอีก 16 ไฟล์ (ผลต่อ OCR ยังไม่ได้วัด จึงไม่เปิดเป็นค่าตั้งต้น)
STREAK_GRAY = 230  # เทาจางกว่านี้นับเป็นเส้น — Otsu จับเส้นจางไม่ได้ (ที่มาเดียวกัน)
STREAK_KEEP_DARK = 110  # พิกเซลในแถบที่มืดกว่านี้คือข้อความที่เส้นพาดผ่าน ไม่ลบ (ที่มาเดียวกัน)
STREAK_CLOSE = (41 / 1632, 7 / 1632)  # kernel closing ต่อจุดเป็นแถบ (สัดส่วนของความกว้างหน้า)
STREAK_EDGE = 0.03  # ต้องแตะทั้งขอบซ้ายและขวาภายในเท่านี้ของความกว้าง (เส้นตารางจริงไม่ถึงขอบภาพดิบ)
STREAK_MAX_H = 0.02  # แถบต้องเตี้ยกว่าเท่านี้ของความสูงหน้า


def streak_mask(img: np.ndarray) -> np.ndarray:
    """แถบเส้นสกปรกจากเครื่องสแกน (255 = เส้น) — ใช้ได้กับ **ภาพดิบ** เท่านั้น:
    หลังหมุน/deskew มุมภาพเติมขาว เส้นจึงไม่แตะขอบ และหลัง crop เส้นตารางจริงแตะขอบได้ (ลบผิด)"""
    gray = to_gray(img)
    h, w = gray.shape
    light = ((gray < STREAK_GRAY) * 255).astype(np.uint8)
    kx, ky = (max(1, round(f * w)) for f in STREAK_CLOSE)
    joined = cv2.morphologyEx(light, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)))
    long_ = cv2.morphologyEx(joined, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (w // 3, 1)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(long_, connectivity=8)
    out = np.zeros_like(gray)
    for i in range(1, n):
        x, _, bw, bh, _ = stats[i]
        if x <= STREAK_EDGE * w and x + bw >= (1 - STREAK_EDGE) * w and bh < STREAK_MAX_H * h:
            out[labels == i] = 255
    return out


def remove_streaks(img: np.ndarray) -> tuple[np.ndarray, int]:
    """ระบายขาวแถบเส้นสแกนเฉพาะพิกเซลที่สว่างกว่า ``STREAK_KEEP_DARK`` (ข้อความดำที่เส้นพาดผ่านยังอยู่) คืน (ภาพ, จำนวนเส้น)."""
    band = streak_mask(img)
    n = cv2.connectedComponents(band)[0] - 1
    if n == 0:
        return img, 0
    out = img.copy()
    out[(band > 0) & (to_gray(img) > STREAK_KEEP_DARK)] = 255
    return out, int(n)


def denoise(img: np.ndarray, method: str = "median") -> np.ndarray:
    """ลด noise จากการสแกน (จุดดำ/เกรน JPEG).

    - median    : เร็ว เหมาะกับ salt-and-pepper
    - nlmeans   : คุณภาพดีสุด แต่ช้ากว่า
    - bilateral : ลด noise แต่รักษาขอบตัวอักษร
    """
    if method in (None, "none"):
        return img
    if method == "median":
        return cv2.medianBlur(img, 3)
    if method == "nlmeans":
        if img.ndim == 3:
            return cv2.fastNlMeansDenoisingColored(img, None, 7, 7, 7, 21)
        return cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)
    if method == "bilateral":
        return cv2.bilateralFilter(img, 9, 50, 50)
    raise ValueError(f"unknown denoise method: {method}")
