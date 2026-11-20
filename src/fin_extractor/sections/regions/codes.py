"""หา QR/บาร์โค้ด ถอดรหัสเก็บใน manifest แล้วระบายขาวก่อนแบ่ง region

คัดลอกจาก learning/preprocess_finance (src/finance_ocr/preprocess/mask.py) ตรงตัว ค่าคงที่วัดบนสแกนชุดเดียวกัน 18 ใบที่ 200 dpi — ใช้กับภาพที่ย่อเป็น 200 dpi เท่านั้น (ดู run.py)

Find non-text graphics (QR codes, barcodes) and blank them out before OCR.

OpenCV's QR detector alone misses most codes on these scans (it found 3 of 7), so
candidates come from a layout heuristic and the detector is only used to *decode*:
  qr      - square-ish block whose ink density is far above text (text ~0.25-0.37, QR ~0.55)
  barcode - wide block dominated by vertical edges (text ~1-1.7, barcodes ~5.5)
Decoded payloads go into the manifest: they are exact, unlike OCR of the same info.
"""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import Box, pad_box
from .masks import ink_mask

_qr = cv2.QRCodeDetector()
_bar = cv2.barcode.BarcodeDetector() if hasattr(cv2, "barcode") else None


def _decode(img: np.ndarray, kind: str) -> str:
    """Try to decode a candidate crop; returns '' when unreadable."""
    roi = cv2.copyMakeBorder(img, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    for scale in (1.0, 2.0):
        r = roi if scale == 1 else cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        try:
            if kind == "qr":
                data, _, _ = _qr.detectAndDecode(r)
            elif _bar is not None:
                data = _bar.detectAndDecode(r)[0] or ""  # (text, points, straight_code)
            else:
                data = ""
        except cv2.error:
            data = ""
        if data:
            return data
    return ""


def _bar_rows(roi: np.ndarray, y: int, bh: int) -> tuple[int, int]:
    """Shrink a barcode box vertically to the rows made of bars.

    Bar rows have many ink/paper transitions; the caption text row has far fewer.
    """
    transitions = np.abs(np.diff((roi > 0).astype(np.int8), axis=1)).sum(axis=1)
    rows = np.flatnonzero(transitions >= 0.6 * transitions.max())
    if rows.size == 0:
        return y, bh
    return y + int(rows[0]), int(rows[-1] - rows[0] + 1)


def find_codes(img: np.ndarray, cfg: dict) -> list[dict]:
    mask = ink_mask(img)
    h, w = mask.shape
    blobs = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(blobs, connectivity=8)
    gx = np.abs(cv2.Sobel(mask, cv2.CV_32F, 1, 0))
    gy = np.abs(cv2.Sobel(mask, cv2.CV_32F, 0, 1))
    found = []
    for i in range(1, n):
        x, y, bw, bh, _ = stats[i]
        if bw < cfg["min_size_frac"] * w or bh < cfg["min_size_frac"] * w * 0.4 or bw > 0.5 * w:
            continue
        density = float((mask[y:y + bh, x:x + bw] > 0).mean())
        vert_ratio = float(gx[y:y + bh, x:x + bw].sum() / (gy[y:y + bh, x:x + bw].sum() + 1))
        aspect = bw / bh
        if 0.7 <= aspect <= 1.4 and density >= cfg["qr_min_density"] and bw >= cfg["qr_min_size_frac"] * w:
            kind = "qr"
        elif aspect >= 2.5 and vert_ratio >= cfg["barcode_min_vert_ratio"] and density >= 0.3:
            kind = "barcode"
        else:
            continue
        if kind == "barcode":  # keep the human-readable number printed under/over the bars
            y, bh = _bar_rows(mask[y:y + bh, x:x + bw], y, bh)
        box = (int(x), int(y), int(x + bw), int(y + bh))
        found.append({"kind": kind, "box": box, "data": _decode(img[y:y + bh, x:x + bw], kind),
                      "density": round(density, 2), "vert_ratio": round(vert_ratio, 2)})
    return found


def blank(img: np.ndarray, boxes: list[Box], pad_px: int = 6) -> np.ndarray:
    out = img.copy()
    for b in boxes:
        x0, y0, x1, y1 = pad_box(b, img.shape, pad_px)
        out[y0:y1, x0:x1] = 255
    return out
