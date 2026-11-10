"""สร้างใบกำกับภาษีไทยสังเคราะห์ที่ "รู้เฉลย" เพื่อวัด preprocessing

ใบเดียวกัน (เนื้อหาเดิม) ถูกทำให้เสียต่างกันทีละปัจจัย: มุมเอียง, สีตัวอักษร, สภาพการสแกน
จึงเทียบได้ว่าปัจจัยไหนทำให้ OCR พลาด เฉลยเก็บใน ``truth.json`` (มุมเอียงจริง สีจริง และค่าเงินจริง)

ข้อมูลทั้งหมดเป็นข้อมูลสมมติ (เลขผู้เสียภาษีสร้างจากสูตร checksum ไม่ใช่ของใคร)
ไม่ใช่สแกนจริง ใช้เทียบ "ปัจจัยต่อปัจจัย" ได้ แต่ไม่แทนการทดสอบกับสแกนจริงของผู้ใช้

    .venv/Scripts/python scripts/make_synthetic_samples.py [--out data/raw/synthetic]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "C:/Windows/Fonts/tahoma.ttf"
FONT_BOLD = "C:/Windows/Fonts/tahomabd.ttf"
W, H = 1654, 2339  # A4 ที่ 200 dpi

# สี (R, G, B) ที่พบได้จริงในเอกสาร: หมึกดำ, ปากกา/เครื่องพิมพ์สีน้ำเงิน, ตรายางแดง, หมึกซีด/สีอ่อน
INKS = {
    "black": (20, 20, 20),
    "blue": (30, 60, 170),
    "red": (200, 30, 30),
    "orange_light": (240, 160, 90),
    "gray_faded": (170, 170, 170),
}


# หมุนตามเข็มนาฬิกา (0 = ไม่หมุน)
ROT_CODES = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def tax_id() -> str:
    """เลขผู้เสียภาษี 13 หลักที่ checksum ถูกต้อง (ข้อมูลสมมติ)"""
    d = [0, 1, 0, 5, 5, 6, 1, 2, 3, 4, 5, 6]
    check = (11 - sum(x * (13 - i) for i, x in enumerate(d)) % 11) % 10
    return "".join(map(str, d)) + str(check)


ITEMS = [
    ("ค่าบริการวิเคราะห์ข้อมูล", 3, 4500.00),
    ("ค่าติดตั้งระบบรายงาน", 1, 12800.00),
    ("ค่าฝึกอบรมผู้ใช้งาน", 2, 3200.00),
]


def truth_values() -> dict:
    subtotal = round(sum(q * p for _, q, p in ITEMS), 2)
    vat = round(subtotal * 0.07, 2)
    return {
        "seller_name": "บริษัท ตัวอย่างเทคโนโลยี จำกัด",
        "seller_tax_id": tax_id(),
        "invoice_no": "INV-2568-00127",
        "date": "18/05/2568",
        "subtotal": subtotal,
        "vat": vat,
        "total": round(subtotal + vat, 2),
    }


def render_page(ink: tuple[int, int, int], header_band: bool, stamp: bool) -> Image.Image:
    """วาดใบกำกับภาษี: ตัวอักษรเนื้อหาใช้สี ink ส่วนเส้นตารางเป็นสีดำเสมอ"""
    t = truth_values()
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    f = lambda s: ImageFont.truetype(FONT, s)  # noqa: E731
    fb = lambda s: ImageFont.truetype(FONT_BOLD, s)  # noqa: E731
    line = (0, 0, 0)

    if header_band:  # แถบหัวสีพื้นสีอ่อนตัวอักษรขาว/เข้ม ตัดกับพื้นสีไม่ใช่พื้นขาว
        d.rectangle([80, 80, W - 80, 230], fill=(70, 130, 180))
        d.text((110, 110), "ใบกำกับภาษี / TAX INVOICE", font=fb(54), fill=(255, 255, 255))
    else:
        d.text((110, 110), "ใบกำกับภาษี / TAX INVOICE", font=fb(54), fill=ink)

    d.text((110, 270), t["seller_name"], font=fb(40), fill=ink)
    d.text((110, 330), "123 ถนนตัวอย่าง แขวงทดสอบ เขตสมมติ กรุงเทพมหานคร 10110", font=f(32), fill=ink)
    d.text((110, 380), f"เลขประจำตัวผู้เสียภาษี {t['seller_tax_id']}", font=f(34), fill=ink)
    d.text((1000, 270), f"เลขที่ {t['invoice_no']}", font=f(36), fill=ink)
    d.text((1000, 330), f"วันที่ {t['date']}", font=f(36), fill=ink)
    d.text((110, 470), "ลูกค้า: บริษัท ผู้ซื้อสมมติ จำกัด", font=f(34), fill=ink)

    # ตาราง
    x = [110, 160, 900, 1060, 1280, W - 110]
    y0, rh = 560, 80
    d.rectangle([x[0], y0, x[-1], y0 + rh * (len(ITEMS) + 1 + 3)], outline=line, width=3)
    for xx in x[1:-1]:
        d.line([xx, y0, xx, y0 + rh * (len(ITEMS) + 1)], fill=line, width=2)
    heads = ["#", "รายการ", "จำนวน", "ราคา/หน่วย", "จำนวนเงิน"]
    for i, h in enumerate(heads):
        d.text((x[i] + 12, y0 + 18), h, font=fb(32), fill=ink)
    for r in range(len(ITEMS) + 2):
        d.line([x[0], y0 + rh * (r + 1), x[-1], y0 + rh * (r + 1)], fill=line, width=2)
    for r, (name, q, p) in enumerate(ITEMS, start=1):
        yy = y0 + rh * r + 18
        d.text((x[0] + 14, yy), str(r), font=f(32), fill=ink)
        d.text((x[1] + 12, yy), name, font=f(32), fill=ink)
        d.text((x[2] + 12, yy), str(q), font=f(32), fill=ink)
        d.text((x[3] + 12, yy), f"{p:,.2f}", font=f(32), fill=ink)
        d.text((x[4] + 12, yy), f"{q * p:,.2f}", font=f(32), fill=ink)
    base = y0 + rh * (len(ITEMS) + 1)
    for k, (label, val) in enumerate(
        [("รวมเป็นเงิน", t["subtotal"]), ("ภาษีมูลค่าเพิ่ม 7%", t["vat"]), ("จำนวนเงินรวมทั้งสิ้น", t["total"])]
    ):
        yy = base + rh * k + 18
        d.text((x[2] + 12, yy), label, font=fb(32), fill=ink)
        d.text((x[4] + 12, yy), f"{val:,.2f}", font=fb(32), fill=ink)

    if stamp:  # ตรายางสีแดงทับบางส่วนของข้อความ
        st = Image.new("RGBA", (520, 220), (0, 0, 0, 0))
        sd = ImageDraw.Draw(st)
        sd.rounded_rectangle([6, 6, 514, 214], radius=24, outline=(200, 30, 30, 200), width=8)
        sd.text((40, 40), "ชำระเงินแล้ว", font=ImageFont.truetype(FONT_BOLD, 64), fill=(200, 30, 30, 200))
        sd.text((40, 130), "PAID 18/05/2568", font=ImageFont.truetype(FONT_BOLD, 40), fill=(200, 30, 30, 200))
        st = st.rotate(12, expand=True, resample=Image.BICUBIC)
        im.paste(st, (950, 1000), st)
    return im


def scan_effects(img: np.ndarray, rng: np.random.Generator, level: str) -> np.ndarray:
    """เลียนแบบสแกน: กระดาษอมเหลือง, แสงไม่สม่ำเสมอ, เบลอเล็กน้อย, เกรน, JPEG"""
    out = img.astype(np.float32)
    if level == "none":
        return img
    gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)
    light = 1.0 - 0.18 * (gx / W) - 0.10 * (gy / H)  # ขอบขวาล่างมืดกว่า
    out *= light[..., None]
    out *= np.array([0.93, 0.97, 1.0], dtype=np.float32)  # กระดาษอมเหลืองเล็กน้อย (BGR)
    out = cv2.GaussianBlur(out, (0, 0), 1.0)
    out += rng.normal(0, 6, out.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def skew(img: np.ndarray, angle: float) -> np.ndarray:
    """หมุนทวนเข็ม ``angle`` องศาบน canvas ขนาดเดิม เติมพื้นขาวอมเทา เหมือนขอบกระดาษบนสแกนเนอร์"""
    if angle == 0:
        return img
    m = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (W, H), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(235, 235, 235))


def variants() -> list[dict]:
    """แต่ละตัวแปรเปลี่ยนปัจจัยเดียวจาก baseline เพื่อให้เทียบกันตรง ๆ"""
    base = dict(ink="black", angle=0.0, band=False, stamp=False, scan="none", rot=0)
    v = [dict(base, name="00_baseline")]
    for a in (-4.0, -1.5, 2.0, 6.0):
        v.append(dict(base, name=f"skew_{a:+.1f}", angle=a))
    for ink in ("blue", "red", "orange_light", "gray_faded"):
        v.append(dict(base, name=f"ink_{ink}", ink=ink))
    v.append(dict(base, name="band_colored_header", band=True))
    v.append(dict(base, name="stamp_red_overlap", stamp=True))
    v.append(dict(base, name="scan_effects_only", scan="real"))
    # ภาพที่สแกนตะแคง/กลับหัว (หมุนตามเข็ม rot องศา) ต้องหมุนกลับก่อนอ่าน
    for r in (90, 180, 270):
        v.append(dict(base, name=f"rot_{r}", rot=r))
    v.append(dict(base, name="combo_blue_rot90_skew+3_scan", ink="blue", angle=3.0, scan="real", rot=90))
    v.append(dict(base, name="combo_orange_rot180_skew-2_scan", ink="orange_light", angle=-2.0, scan="real", rot=180))
    v.append(dict(base, name="combo_faded_rot270_skew+5_stamp_scan", ink="gray_faded", angle=5.0, stamp=True, scan="real", rot=270))
    # ของจริงมักเจอหลายปัจจัยพร้อมกัน
    v.append(dict(base, name="combo_blue_skew+3_scan", ink="blue", angle=3.0, scan="real"))
    v.append(dict(base, name="combo_orange_skew-2_scan", ink="orange_light", angle=-2.0, scan="real"))
    v.append(dict(base, name="combo_faded_skew+5_stamp_scan", ink="gray_faded", angle=5.0, stamp=True, scan="real"))
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/synthetic")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)  # ตายตัว เพื่อให้ได้ภาพเดิมทุกครั้ง

    truth = {"fields": truth_values(), "files": {}}
    for v in variants():
        page = render_page(INKS[v["ink"]], v["band"], v["stamp"])
        img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
        img = skew(img, v["angle"])
        img = scan_effects(img, rng, v["scan"])
        if v["rot"]:
            img = cv2.rotate(img, ROT_CODES[v["rot"]])
        path = out / f"{v['name']}.jpg"
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 80 if v["scan"] != "none" else 95])
        truth["files"][path.name] = {
            "skew_applied_ccw_deg": v["angle"],
            "rotate_applied_cw_deg": v["rot"],
            "ink": v["ink"], "ink_rgb": INKS[v["ink"]],
            "colored_header": v["band"], "stamp": v["stamp"], "scan_effects": v["scan"] != "none",
        }
        print("wrote", path)
    (out / "truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
