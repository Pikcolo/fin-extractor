"""ภาพตรวจด้วยตา: overlay region/ตาราง และ contact sheet

คัดลอกจาก learning/preprocess_finance (src/finance_ocr/debug/viz.py) ตรงตัว ค่าคงที่วัดบนสแกนชุดเดียวกัน (ชุดสแกนทดสอบ) ที่ 200 dpi — ใช้กับภาพที่ย่อเป็น 200 dpi เท่านั้น (ดู run.py)

Debug artefacts: per-stage snapshots, region overlays, contact sheets."""
from __future__ import annotations


import cv2
import numpy as np

PALETTE = [(220, 50, 50), (40, 160, 40), (30, 90, 220), (200, 120, 0), (160, 40, 160),
           (0, 150, 170), (120, 120, 0), (90, 60, 200), (0, 100, 255), (180, 0, 90)]


def draw_regions(img: np.ndarray, regions: list[dict], frame=None, hlines=None) -> np.ndarray:
    out = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    t = max(2, out.shape[1] // 500)
    if frame is not None:
        cv2.rectangle(out, frame[:2], frame[2:], (128, 128, 128), t, lineType=cv2.LINE_4)
    for l in hlines or []:  # detected rules, dashed look via thin magenta
        cv2.line(out, (l["x0"], l["y"]), (l["x1"], l["y"]), (255, 0, 255), 1)
    for i, r in enumerate(regions):
        color = PALETTE[i % len(PALETTE)]
        x0, y0, x1, y1 = r["box"]
        overlay = out.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
        out = cv2.addWeighted(overlay, 0.12, out, 0.88, 0)
        cv2.rectangle(out, (x0, y0), (x1, y1), color, t)
        cv2.putText(out, r["name"], (x0 + 6, y0 + 14 * t), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * t, color, t, cv2.LINE_AA)
    return out


def draw_tables(overlay: np.ndarray, tables: list[dict], vlines: list[dict]) -> np.ndarray:
    """Vertical rules (orange), header cells (red), value cells (green, trimmed box)."""
    out = overlay.copy()
    for l in vlines:
        cv2.line(out, (l["x"], l["y0"]), (l["x"], l["y1"]), (0, 140, 255), 1)
    for t in tables:
        for r in t["rows"]:
            color = (0, 0, 220) if r["kind"] == "header" else (0, 160, 0)
            for c in r["cells"]:
                if c["trim"] is not None:
                    x0, y0, x1, y1 = c["trim"]
                    cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
    return out


def contact_sheet(images: list[tuple[str, np.ndarray]], cols: int = 6, thumb_w: int = 400) -> np.ndarray:
    thumbs = []
    for label, img in images:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        s = thumb_w / img.shape[1]
        th = cv2.resize(img, (thumb_w, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
        th = cv2.copyMakeBorder(th, 28, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(th, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200), 1, cv2.LINE_AA)
        thumbs.append(th)
    cell_h = max(t.shape[0] for t in thumbs)
    cell_w = thumbs[0].shape[1]
    rows = (len(thumbs) + cols - 1) // cols
    sheet = np.full((rows * cell_h, cols * cell_w, 3), 255, np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet[r * cell_h:r * cell_h + t.shape[0], c * cell_w:(c + 1) * cell_w] = t
    return sheet
