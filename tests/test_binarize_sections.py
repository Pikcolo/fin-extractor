"""ภาพย่อยแบบ bin (images.binarize_inverted / drop_empty_rows_by_table): ตัวขาวบนดำ ไม่ขึ้นกับสีพื้น และลบแถวตารางที่ช่องค่าว่าง."""

import cv2
import numpy as np

from fin_extractor.sections import images


def _text(img: np.ndarray, s: str, org: tuple[int, int], color) -> None:
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)


def test_binarize_is_white_text_on_black_for_any_fill():
    paper = np.full((200, 600, 3), 255, np.uint8)
    _text(paper, "2,000.00", (20, 100), (0, 0, 0))
    green = np.full((200, 600, 3), (60, 170, 60), np.uint8)  # แถบเขียว ตัวขาว
    _text(green, "2,000.00", (20, 100), (255, 255, 255))
    for img in (paper, green):
        b = images.binarize_inverted(img)
        assert set(np.unique(b)) <= {0, 255}
        assert b[:20].mean() == 0  # พื้นดำ
        assert 0 < (b > 0).mean() < 0.2  # ตัวอักษรเป็นขาว ส่วนน้อย


def _form(with_value_in_row2: bool) -> np.ndarray:
    """ตารางฟอร์ม: ป้ายซ้าย กว้าง + 3 คอลัมน์ค่า 4 แถว มีเลขเฉพาะแถว 1 (และแถว 2 ถ้าสั่ง) แถวอื่นมีแต่จุดไข่ปลา."""
    img = np.full((800, 1200, 3), 255, np.uint8)
    xs = [40, 600, 800, 1000, 1160]
    for x in xs:
        cv2.line(img, (x, 40), (x, 760), (0, 0, 0), 3)
    for r in range(4):
        y = 100 + r * 180
        _text(img, f"{r + 1}. label row {r + 1}", (60, y), (0, 0, 0))
        for x0, x1 in zip(xs[1:], xs[2:] + [xs[-1]]):
            if x0 < x1:
                for x in range(x0 + 10, x1 - 10, 12):
                    cv2.circle(img, (x, y + 8), 1, (0, 0, 0), -1)
        if r == 0 or (r == 1 and with_value_in_row2):
            _text(img, "2000", (xs[2] + 20, y), (0, 0, 0))
    return img


def test_drop_empty_rows_keeps_only_rows_with_values():
    b = images.binarize_inverted(_form(False))
    sec = {"id": "s00", "table": 0}
    (out, info), = images.drop_empty_rows_by_table([(sec, b)])
    assert info["applied"]
    assert out[60:130, 60:500].any()  # ป้ายของแถวที่มีค่า
    assert out[90:130, 820:980].any()  # ตัวเลข
    assert not out[240:500, 60:560].any()  # ป้ายของแถวที่ไม่มีค่า หายไป
    assert not out[600:700, 60:560].any()


def test_drop_empty_rows_leaves_tables_without_any_value():
    img = _form(False)
    img[60:130, 800:1000] = 255  # ลบเลขทิ้ง
    b = images.binarize_inverted(img)
    (out, info), = images.drop_empty_rows_by_table([({"id": "s00", "table": 0}, b)])
    assert not info["applied"] and (out == b).all()
