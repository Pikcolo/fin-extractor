"""ส่วนที่ไม่ส่งเข้า OCR: ระบาย QR/บาร์โค้ดก่อนแบ่ง section (ข้อความในรหัสถอดเก็บไว้ในผังแทน ตรงตัวกว่า OCR)
และ (ทางเลือก ``trim_sections``) หดกรอบแต่ละส่วนให้เหลือแค่ข้อความ ตัดพื้นว่างทิ้ง

ใช้ตัวตรวจของโหมด regions (``regions/codes.py``) กับค่า ``mask`` / ``ref_dpi`` ใน configs/regions/pipeline.yaml ตัวเดียวกัน (วัดบนสแกน 18 ใบที่ 200 dpi)
ผลบน 18 ใบ: เจอ QR/บาร์โค้ดครบ 8/8 (1312 QR สองก้อน · 1313/1314 QR สอง + บาร์โค้ดหนึ่ง ถอดได้ทุกก้อน) และตราครุฑบนหัว 1316 หนึ่งก้อน
(ถอดไม่ได้ ดูแล้วเป็นภาพ ไม่ใช่ข้อความ ระบายได้ไม่เสียข้อมูล)
โลโก้/ตรายางทั่วไปไม่ถูกตรวจ: วัดชิ้นหมึกที่สูง ≥ 35 px บน 18 ใบแล้ว ชิ้นของโลโก้ (1308 1312 1311) มีขนาดและสีอยู่ช่วงเดียวกับลายมือปากกาน้ำเงิน
ของ 1320/1322 (ชื่อผู้รับ จำนวนเงิน = ข้อมูล) เกณฑ์ขนาด/สีที่ตัดโลโก้จะตัดลายมือไปด้วย
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from .regions import codes

CONFIG = Path(__file__).resolve().parents[3] / "configs" / "regions" / "pipeline.yaml"
PAD_PX = 6  # ขยายกล่องเท่า ``codes.blank`` (px ที่ ref_dpi)


def _region_cfg() -> dict:
    with open(CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _mask_cfg() -> tuple[float, dict]:
    cfg = _region_cfg()
    return float(cfg["ref_dpi"]), cfg["mask"]


TRIM_MODES = ("all", "text-lr")  # all = ทุกส่วน ทั้งสี่ด้าน · text-lr = เฉพาะส่วนข้อความ (kind text) เฉพาะซ้าย/ขวา ไม่แตะตาราง


def trim_sections(full: np.ndarray, work: np.ndarray, secs: list[dict], src_dpi: float, mode: str = "all") -> None:
    """หดกรอบของส่วนที่เก็บ (``keep``) ให้เหลือแค่ข้อความ + ขอบ ตัดพื้นว่างที่ไม่มีข้อความทิ้ง ไม่ส่งเข้า OCR
    ``mode``: ``all`` = ทุกส่วนทั้งสี่ด้าน · ``text-lr`` = เฉพาะส่วนข้อความ หดแค่ซ้าย/ขวา (บน/ล่างคงตามแถบ ตารางไม่แตะ: เส้นขอบตารางยังอยู่ครบ)
    ใช้ ``regions/geometry.trim_box`` ตัวเดียวกับโหมด regions บนหน้ากากข้อความระดับหน้า (เส้นตารางและจุดเล็กไม่นับ) ที่ ref_dpi
    และ ``export.trim_pad_px`` ของ configs/regions/pipeline.yaml (ค่าที่วัดบน 18 ใบที่ 200 dpi ไม่มีค่าใหม่)
    กรอบเดิมเก็บใน ``band`` (ลำดับอ่านและ id ไม่เปลี่ยน) ส่วนที่หาข้อความไม่เจอหลังลบเส้น (เช่น เหลือแต่เส้นตาราง) คงกรอบเดิม."""
    from .regions.geometry import trim_box
    from .regions.masks import ink_mask, text_mask

    if mode not in TRIM_MODES:
        raise ValueError(f"trim mode ต้องเป็น {TRIM_MODES}: {mode!r}")
    cfg = _region_cfg()
    scale = min(1.0, float(cfg["ref_dpi"]) / src_dpi)
    small = cv2.resize(full, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else full
    text = text_mask(ink_mask(small))
    fx = full.shape[1] / work.shape[1] * scale  # สำเนาทำงาน → สำเนา ref_dpi
    fy = full.shape[0] / work.shape[0] * scale
    for s in secs:
        if not s["keep"] or (mode == "text-lr" and s["kind"] != "text"):
            continue
        x, y, w, h = s["bbox"]
        box = (int(x * fx), int(y * fy), int(np.ceil((x + w) * fx)), int(np.ceil((y + h) * fy)))
        t = trim_box(text, box, cfg["export"]["trim_pad_px"])
        if t is None:
            continue
        nx0, ny0 = max(x, int(t[0] / fx)), max(y, int(t[1] / fy))
        nx1, ny1 = min(x + w, int(np.ceil(t[2] / fx))), min(y + h, int(np.ceil(t[3] / fy)))
        if mode == "text-lr":
            ny0, ny1 = y, y + h
        if (nx0, ny0, nx1 - nx0, ny1 - ny0) != (x, y, w, h) and nx1 > nx0 and ny1 > ny0:
            s["band"] = [x, y, w, h]
            s["bbox"] = [nx0, ny0, nx1 - nx0, ny1 - ny0]


def find_codes(full: np.ndarray, src_dpi: float) -> list[dict]:
    """QR/บาร์โค้ดบนภาพ final ความละเอียดเต็ม → ``[{kind, data, box: [x0, y0, x1, y1]}]`` พิกัดภาพเต็ม (รวมขอบ ``PAD_PX`` แล้ว)
    ตรวจบนสำเนาที่ย่อเป็น ref_dpi (ค่าคงที่ของตัวตรวจเป็น px ที่ความละเอียดนั้น) ไม่ขยายภาพที่เล็กกว่า."""
    ref_dpi, cfg = _mask_cfg()
    scale = min(1.0, ref_dpi / src_dpi)
    small = cv2.resize(full, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else full
    h, w = full.shape[:2]
    pad = PAD_PX / scale
    out = []
    for c in codes.find_codes(small, cfg):
        x0, y0, x1, y1 = c["box"]
        out.append({"kind": c["kind"], "data": c["data"],
                    "box": [max(0, int(x0 / scale - pad)), max(0, int(y0 / scale - pad)),
                            min(w, int(np.ceil(x1 / scale + pad))), min(h, int(np.ceil(y1 / scale + pad)))]})
    return out


def blank(full: np.ndarray, found: list[dict]) -> np.ndarray:
    """ระบายขาวทุกกล่องใน ``found`` (พิกัดภาพเต็ม) คืนสำเนา ไม่มีอะไรให้ระบาย = ภาพเดิม."""
    if not found:
        return full
    out = full.copy()
    for c in found:
        x0, y0, x1, y1 = c["box"]
        out[y0:y1, x0:x1] = 255
    return out


def mark_sections(secs: list[dict], found: list[dict], full_shape: tuple[int, ...], work_shape: tuple[int, ...]) -> None:
    """ใส่ ``masked`` (ชนิดของรหัสที่ถูกระบายในส่วนนั้น) ให้ส่วนที่ทับกล่องที่ระบาย — ลายนิ้วมือผังของขั้น OCR นับคีย์นี้
    ผล OCR เก่าที่อ่านภาพก่อนระบายจึงไม่ถูกใช้ต่อ (ส่วนที่ไม่ทับไม่มีคีย์นี้ ผลเดิมยังใช้ได้)."""
    fx, fy = work_shape[1] / full_shape[1], work_shape[0] / full_shape[0]
    for s in secs:
        x, y, w, h = s["bbox"]
        kinds = sorted({c["kind"] for c in found
                        if c["box"][0] * fx < x + w and c["box"][2] * fx > x and c["box"][1] * fy < y + h and c["box"][3] * fy > y})
        if kinds:
            s["masked"] = kinds
