"""วัด preprocessing กับชุดสังเคราะห์ (``make_synthetic_samples.py``) ที่รู้เฉลย

วัดสามเรื่อง ผลบันทึกใน ``--out`` (ค่าตั้งต้น ``data/benchmarks/preprocessing_synthetic.json``)
  1. orientation : Tesseract OSD เทียบกับตัวจำแนกทิศทั้งหน้าของ Paddle ถูกกี่ใบจากภาพตะแคง
  2. deskew      : มุมที่วัดได้คลาดจากมุมที่ใส่จริงกี่องศา
  3. ocr         : Paddle อ่านค่าสำคัญได้ครบกี่ค่า (จาก 6) เมื่อเปลี่ยนชุด preprocessing

ค่าสำคัญ 6 ค่า = เลขผู้เสียภาษี, เลขที่ใบกำกับ, วันที่, รวมเป็นเงิน, VAT, ยอดรวม
นับว่า "อ่านเจอ" เมื่อข้อความ OCR (ตัดช่องว่างออก) มีค่านั้นอยู่

    .venv/Scripts/python scripts/benchmark_preprocessing.py [--samples data/raw/synthetic] [--skip-ocr] [--out ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fin_extractor.config import load_config  # noqa: E402
from fin_extractor.paths import DataPaths  # noqa: E402
from fin_extractor.preprocessing import geometry  # noqa: E402
from fin_extractor.preprocessing.pipeline import preprocess  # noqa: E402

ROT_CODES = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def read(path: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def key_values(fields: dict) -> list[str]:
    return [
        fields["seller_tax_id"], fields["invoice_no"], fields["date"],
        f"{fields['subtotal']:,.2f}", f"{fields['vat']:,.2f}", f"{fields['total']:,.2f}",
    ]


def found(text: str, fields: dict) -> int:
    flat = "".join(text.split())
    return sum("".join(v.split()) in flat for v in key_values(fields))


def upright(img: np.ndarray, rot_cw: int) -> np.ndarray:
    """หมุนภาพกลับตามเฉลย เพื่อแยกวัด OCR ออกจากเรื่องทิศ"""
    return img if rot_cw == 0 else cv2.rotate(img, ROT_CODES[(360 - rot_cw) % 360])


# ---------- 1. orientation ----------

def paddle_orientation_model():
    from fin_extractor.ocr.paddle_engine import _expose_nvidia_dlls

    _expose_nvidia_dlls()
    from paddlex import create_model

    return create_model(model_name="PP-LCNet_x1_0_doc_ori")


def bench_orientation(files: dict, root: Path) -> list[dict]:
    model = paddle_orientation_model()
    rows = []
    for name, meta in files.items():
        img = read(root / name)
        rot = meta["rotate_applied_cw_deg"]
        fix = (360 - rot) % 360  # หมุนตามเข็มกี่องศาถึงจะตั้งตรง
        osd = geometry.detect_orientation(img)
        t = time.perf_counter()
        label = int(next(iter(model.predict(img, batch_size=1)))["label_names"][0])
        rows.append({
            "file": name, "applied_cw": rot, "expected_fix_cw": fix,
            "osd_fix_cw": osd, "osd_ok": osd == fix,
            # ป้ายของ Paddle คือ "มุมที่ภาพถูกหมุนอยู่" ไม่ใช่มุมแก้ — เก็บค่าดิบไว้ เทียบสองทิศทาง
            "paddle_label": label,
            "paddle_ok_if_label_is_applied_cw": label == rot,
            "paddle_ok_if_label_is_applied_ccw": label == (360 - rot) % 360,
            "paddle_ms": round((time.perf_counter() - t) * 1000),
        })
    return rows


# ---------- 2. deskew ----------

def bench_deskew(files: dict, root: Path) -> list[dict]:
    rows = []
    for name, meta in files.items():
        if meta["rotate_applied_cw_deg"]:
            continue  # วัดมุมเอียงบนภาพที่ตั้งตรงแล้วเท่านั้น
        img = read(root / name)
        applied = meta["skew_applied_ccw_deg"]
        est = geometry.estimate_skew(img, 15.0)
        # estimate_skew คืนมุมที่ต้องหมุนทวนเข็มเพื่อแก้ ดังนั้นควรได้ -applied
        rows.append({"file": name, "applied_ccw": applied, "estimated_fix_ccw": est, "error_deg": round(est + applied, 2)})
    return rows


# ---------- 3. OCR ----------

def variants(cfg: dict) -> dict[str, dict]:
    # ปิด orientation เพราะวัดแยกแล้ว (ส่งภาพที่หมุนกลับตามเฉลยเข้ามา) เหลือเทียบเฉพาะการแต่งภาพ
    base = dict(cfg["preprocessing"], fix_orientation=False)
    return {
        "neural_full": dict(base, denoise="nlmeans", clahe=True),  # ค่าตั้งต้นเดิม: สี + NL-means + CLAHE
        "color_only": dict(base, denoise="none", clahe=False, keep_color=True),  # ค่าตั้งต้นปัจจุบัน
        "gray_only": dict(base, denoise="none", clahe=False, keep_color=False),
        "gray_otsu": dict(base, denoise="none", clahe=False, keep_color=False, binarize="otsu"),
    }


def bench_ocr(files: dict, root: Path, cfg: dict) -> list[dict]:
    from fin_extractor.ocr import create_engine

    engine = create_engine("paddle", cfg)
    fields = json.loads((root / "truth.json").read_text(encoding="utf-8"))["fields"]
    engine.recognize(upright(read(root / "00_baseline.jpg"), 0))  # warm-up ไม่นับเวลา
    rows = []
    for vname, pcfg in variants(cfg).items():
        for name, meta in files.items():
            img = upright(read(root / name), meta["rotate_applied_cw_deg"])
            t = time.perf_counter()
            res = preprocess(img, pcfg)
            t_pre = time.perf_counter() - t
            t = time.perf_counter()
            text = "\n".join(line.text for line in engine.recognize(res.image).lines)
            rows.append({
                "variant": vname, "file": name, "found": found(text, fields),
                "skew_angle": res.meta.get("skew_angle"),
                "pre_s": round(t_pre, 2), "ocr_s": round(time.perf_counter() - t, 2),
            })
            print(f"{vname:12s} {name:42s} {rows[-1]['found']}/6", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="data/raw/synthetic")
    ap.add_argument("--skip-ocr", action="store_true")
    ap.add_argument("--out", default=str(DataPaths().benchmarks() / "preprocessing_synthetic.json"))
    args = ap.parse_args()
    root = Path(args.samples)
    files = json.loads((root / "truth.json").read_text(encoding="utf-8"))["files"]
    cfg = load_config()

    result = {"orientation": bench_orientation({k: v for k, v in files.items() if v["rotate_applied_cw_deg"]}, root),
              "deskew": bench_deskew(files, root)}
    if not args.skip_ocr:
        result["ocr"] = bench_ocr(files, root, cfg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
