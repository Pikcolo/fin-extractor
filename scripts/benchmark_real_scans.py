"""วัด preprocessing กับสแกนจริงที่ไม่มีเฉลย (label-free)

เทียบ preprocessing หลายชุดโดยใช้ตัวชี้วัดที่ไม่ต้องรู้คำตอบที่ถูก:
  tax_ids  : จำนวนเลข 13 หลักที่ checksum ผ่าน (เลขที่ผ่านแล้วแทบแน่ใจว่าอ่านถูก)
  triple   : มีชุดตัวเลข a+b=c และ b≈VAT% ของ a ในข้อความหรือไม่ (ใช้ tolerance/VAT จาก config)
  lines    : จำนวนบรรทัดที่ตรวจเจอ   (ตัวชี้วัดอ่อน — ขึ้นได้แม้เป็นขยะ)
  conf     : ความมั่นใจเฉลี่ยของ Paddle (ตัวชี้วัดอ่อน เช่นกัน)
  sec      : เวลาต่อหน้า (preprocess + OCR)

ผลดิบ (รวมข้อความ OCR) บันทึกใน ``--out`` เพื่อเปิดดูย้อนหลังได้

    .venv/Scripts/python scripts/benchmark_real_scans.py [--input data/raw/finance_doc_scan] [--only base,gray]
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fin_extractor.config import load_config  # noqa: E402
from fin_extractor.io import iter_input_files, load_pages  # noqa: E402
from fin_extractor.paths import DataPaths  # noqa: E402
from fin_extractor.ocr.paddle_engine import PaddleEngine  # noqa: E402
from fin_extractor.preprocessing import preprocess  # noqa: E402
from fin_extractor.validation.tax_id import is_valid_thai_tax_id  # noqa: E402

# ชื่อชุด → ค่าที่ override ทับ preprocessing ของ config ("raw" = ไม่ preprocess เลย)
VARIANTS: dict[str, dict | None] = {
    "base": {},
    "raw": None,
    "no_deskew": {"deskew": False},
    "no_orient": {"fix_orientation": False},
    "crop": {"remove_borders": True},
    "gray": {"keep_color": False},
    "median": {"denoise": "median"},
    "nlmeans": {"denoise": "nlmeans"},
    "bilateral": {"denoise": "bilateral"},
    "clahe": {"clahe": True},
    "otsu": {"keep_color": False, "binarize": "otsu"},
    "adaptive": {"keep_color": False, "binarize": "adaptive"},
    "sauvola": {"keep_color": False, "binarize": "sauvola"},
}

NUM_RE = re.compile(r"\d{1,3}(?:,\d{3})*\.\d{2}|\d+\.\d{2}")


def valid_tax_ids(text: str) -> set[str]:
    """หาเลข 13 หลักที่ checksum ผ่าน ยอมให้มีช่องว่าง/ขีดคั่นระหว่างหลัก"""
    found = set()
    for m in re.finditer(r"\d(?:[\s\-]?\d){12}", text):
        digits = re.sub(r"\D", "", m.group())
        if is_valid_thai_tax_id(digits):
            found.add(digits)
    return found


def has_amount_triple(text: str, tol: float, vat: float) -> bool:
    """มีตัวเลข a, b, c ที่ a+b≈c และ b≈vat*a (ก่อนVAT + VAT = รวม)"""
    nums = sorted({float(n.replace(",", "")) for n in NUM_RE.findall(text)})
    nums = [n for n in nums if n > 0]
    for a in nums:
        b_expected = a * vat
        for b in nums:
            if abs(b - b_expected) <= tol + 0.005 * a and any(abs(a + b - c) <= tol for c in nums):
                return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/finance_doc_scan")
    ap.add_argument("--only", help="ชื่อชุดที่ต้องการ คั่นด้วย , (ค่าตั้งต้น: ทุกชุด)")
    ap.add_argument("--out", default=str(DataPaths().benchmarks() / "real_scans.json"))
    args = ap.parse_args()

    cfg = load_config()
    tol = cfg["validation"]["amount_tolerance"]
    vat = cfg["validation"]["default_vat_rate"]
    names = args.only.split(",") if args.only else list(VARIANTS)
    engine = PaddleEngine(**cfg["ocr"]["paddle"])

    files = list(iter_input_files(args.input))
    pages = {p.name: load_pages(p, cfg["loader"]["pdf_dpi"])[0] for p in files}
    results: dict[str, dict] = {n: {} for n in names}

    for name in names:
        override = VARIANTS[name]
        for fname, img in pages.items():
            t = time.perf_counter()
            if override is None:
                proc, meta = img, {}
            else:
                profile = copy.deepcopy(cfg["preprocessing"])
                profile.update(override)
                r = preprocess(img, profile)
                proc, meta = r.image, r.meta
            res = engine.recognize(proc)
            sec = time.perf_counter() - t
            text = res.text
            results[name][fname] = {
                "lines": len(res.lines),
                "conf": res.mean_confidence,
                "tax_ids": sorted(valid_tax_ids(text)),
                "triple": has_amount_triple(text, tol, vat),
                "sec": round(sec, 2),
                "meta": meta,
                "text": text,
            }
        n = len(pages)
        rows = results[name].values()
        print(
            f"{name:10s} tax_ids={sum(len(r['tax_ids']) for r in rows):3d} "
            f"docs_with_taxid={sum(bool(r['tax_ids']) for r in rows):2d}/{n} "
            f"triple={sum(r['triple'] for r in rows):2d}/{n} "
            f"lines={sum(r['lines'] for r in rows) / n:5.1f} "
            f"conf={sum(r['conf'] or 0 for r in rows) / n:.3f} "
            f"sec={sum(r['sec'] for r in rows) / n:.2f}",
            flush=True,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
