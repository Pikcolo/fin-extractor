"""วัดวิธีเตรียมภาพย่อย (section) ก่อนเข้า OCR ด้วย OXR: สีเดิม / เทา luma / luma + พื้นหลังเรียบ / V + พื้นหลังเรียบ

    python -X utf8 scripts/eval_section_preprocess.py 1306 1307 1309 1313 1317

ไม่มีเฉลย จึงใช้ "ตัวเลขการเงินที่ Typhoon และ OpenThai อ่านตรงกันทั้งหน้า" (ยอดเงิน, วันที่, เลขภาษี 13 หลัก) เป็นตัวเทียบ
- recall   = ตัวเลขฉันทามติที่วิธีนั้นอ่านเจอกลับมา (รวมทุกส่วนของใบ) / ฉันทามติทั้งหมด
- spurious = ตัวเลขที่วิธีนั้นอ่านออกมา แต่ทั้งสองตัวอ่านทั้งหน้าไม่มี (อาจแต่งขึ้น หรือเป็นของจริงที่ทั้งสองตัวพลาด แยกไม่ได้)
OXR อ่านตัวเลขกับอังกฤษได้ดี (ไทยไม่ได้) จึงวัดได้แค่ผลของการเตรียมภาพต่อ "ตัวเลข" ไม่ใช่ต่อข้อความไทย
ผมจำกัด max_new_tokens=96 เพราะตัวเลขสั้น และช่องไทยที่วนซ้ำจะได้ไม่เสียเวลา
ผล: data/etc/experiments/section_preprocess/summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notebooks" / "preprocess_lab"))

import labkit as L  # noqa: E402
from fin_extractor.config import load_config  # noqa: E402
from fin_extractor.vlm.oxr_local import OXRLocal  # noqa: E402

DATASET = "finance_doc_scan"
USE = ["สีเดิม", "เทา luma", "luma + พื้นหลังเรียบ", "V + พื้นหลังเรียบ"]
TOKEN = re.compile(r"\d{1,3}(?:,\d{3})*\s?\.\d{2}(?!\d)|\d{2}/\d{2}/\d{4}|(?<!\d)\d(?:[ \-]?\d){12}(?!\d)")


def tokens(text: str) -> set[str]:
    return {re.sub(r"[ ,\-]", "", m.group()) for m in TOKEN.finditer(text)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("docs", nargs="+", help="เลขท้ายชื่อไฟล์ เช่น 1309")
    args = ap.parse_args()
    cfg = load_config()
    paths = L.PATHS
    reader = OXRLocal({**cfg["oxr_local"], "max_new_tokens": 96})
    reader._load()
    result: dict = {}
    for d in args.docs:
        stem = f"BRWAC50DE3689DA_00{d}"
        ty = (paths.raw_data("typhoon", DATASET) / stem / f"{stem}_p1.md").read_text(encoding="utf-8")
        op = (paths.raw_data("openthai", DATASET) / stem / f"{stem}_p1.md").read_text(encoding="utf-8")
        consensus, anyone = tokens(ty) & tokens(op), tokens(ty) | tokens(op)
        read = {v: set() for v in USE}
        secs = {v: 0 for v in USE}
        trunc = {v: 0 for v in USE}
        t_var = {v: 0.0 for v in USE}
        for _, crop in L.iter_section_crops(stem, DATASET):  # ตัดจากภาพทั้งหน้าตามผัง ไม่ใช้ไฟล์ภาพย่อย
            variants = L.clean_variants(crop)
            for v in USE:
                im = variants[v]
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) if im.ndim == 2 else im
                t = time.perf_counter()
                res = reader.recognize(im)
                t_var[v] += time.perf_counter() - t
                read[v] |= tokens(res.markdown)
                trunc[v] += res.truncated
                secs[v] += 1
        result[d] = {"consensus": sorted(consensus), "typhoon_only_or_openthai_only": len(anyone - consensus), "variants": {
            v: {"recall": len(read[v] & consensus), "of": len(consensus), "spurious": len(read[v] - anyone), "truncated": trunc[v],
                "seconds": round(t_var[v], 1), "tokens": sorted(read[v])} for v in USE}}
        print(f"{d}: ฉันทามติ {len(consensus)} ตัว (Typhoon∪OpenThai {len(anyone)})", flush=True)
        for v in USE:
            r = result[d]["variants"][v]
            print(f"    {v:20s} เจอ {r['recall']}/{r['of']} · แปลกปลอม {r['spurious']} · ชนเพดาน {r['truncated']} · {r['seconds']}s", flush=True)
    out = paths.experiments("section_preprocess")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nรวมทุกใบ:")
    for v in USE:
        rec = sum(result[d]["variants"][v]["recall"] for d in result)
        tot = sum(result[d]["variants"][v]["of"] for d in result)
        sp = sum(result[d]["variants"][v]["spurious"] for d in result)
        print(f"  {v:20s} เจอ {rec}/{tot} · แปลกปลอม {sp}")


if __name__ == "__main__":
    main()
