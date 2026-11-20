"""เส้นแบ่งแถวที่เป็นเส้นประ (เช่น ฟอร์ม 50 ทวิ): ``dotted=True`` ต้องตัดตามเส้น และไม่ตัดผ่านข้อความฝั่งป้าย."""

import cv2
import numpy as np

from fin_extractor.sections import layout as L

ROWS = 8
TOP, PITCH = 120, 60


def _form() -> np.ndarray:
    """ตารางกรอบทึบ + เส้นแบ่งคอลัมน์ทึบ; เส้นแบ่งแถวเป็นเส้นประเฉพาะคอลัมน์ขวา; ป้ายแถวอยู่ซ้าย"""
    img = np.full((700, 600), 255, np.uint8)
    x0, x1, split = 30, 570, 330
    y_end = TOP + ROWS * PITCH
    cv2.rectangle(img, (x0, TOP), (x1, y_end), 0, 2)
    cv2.line(img, (split, TOP), (split, y_end), 0, 2)
    for r in range(ROWS):
        cv2.putText(img, f"item {r + 1}", (45, TOP + r * PITCH + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
        y = TOP + (r + 1) * PITCH - 8
        for x in range(split + 4, x1 - 2, 6):  # จุดห่างกัน 3 px
            img[y:y + 2, x:x + 3] = 0
    return img


def test_every_dotted_row_rule_is_found():
    bw = _form()
    rules = L.dotted_rules(L.thin_joined(bw), L.table_regions(bw)[0])
    for r in range(ROWS):
        y = TOP + (r + 1) * PITCH - 8
        assert any(abs(y - ry) <= 3 for ry in rules), f"ไม่เจอเส้นประแถว {r + 1} ที่ y={y}: {rules}"


def test_dotted_cuts_between_rows_and_not_through_text():
    bw = _form()
    plain = L.layout_sections(bw, max_piece_frac=None)
    dotted = L.layout_sections(bw, max_piece_frac=None, dotted=True)
    assert len(dotted) > len(plain)
    ink = L.text_ink(bw)
    for s in dotted:
        for y in (s["bbox"][1], s["bbox"][1] + s["bbox"][3]):
            if 0 < y < bw.shape[0]:
                assert (ink[y, :330] > 0).sum() == 0, f"จุดตัด y={y} พาดข้อความฝั่งป้าย"
