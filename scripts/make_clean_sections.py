"""สร้างภาพย่อย (section) ที่ทำความสะอาดแล้วสำหรับส่งเข้า OCR ไฟล์ละ 3 แบบต่อ section

    python -X utf8 scripts/make_clean_sections.py data/raw/finance_doc_scan
    python -X utf8 scripts/make_clean_sections.py data/raw/finance_doc_scan --only 1309
    python -X utf8 scripts/make_clean_sections.py data/raw/finance_doc_scan --only 1307 --drop-empty-rows

อินพุต: ภาพทั้งหน้าที่ preprocess แล้ว (ตรงแล้ว สี) + ผังแบ่ง section (notebooks/preprocess_lab → features/<ไฟล์>.json)
ผล: data/preprocess/<dataset>/<profile>/sections/clean/<วิธี>/<ไฟล์>/  (วิธี = luma | flat | bin แยกโฟลเดอร์ แต่ละโฟลเดอร์มี sNN.png + index.json ของตัวเอง)
    luma/sNN.png   เทาแบบ luma (ไม่แต่งอะไรอีก) — วัดแล้วไม่แย่กว่าภาพสี และไม่ทำลายตัวอักษรจางบนพื้นสี
    flat/sNN.png   ช่อง V (กำจัดสี) + พื้นหลังเรียบ — ช่วย 1306/1307 แต่ทำลายตัวอักษรจางบนพื้นสีของ 1317
    bin/sNN.png    ขาวดำ **ตัวอักษรขาว กระดาษดำ** ตัดสินจากระยะสีจากพื้นรอบข้าง จึงไม่ขึ้นกับสีพื้น/สีตัวอักษร (ตัวขาวบนแถบสีเขียว/น้ำเงิน ก็ออกมาขาวบนดำ)
                   ใส่ --drop-empty-rows: ในตารางฟอร์ม ทาดำแถวที่ช่องตัวเลขว่าง เหลือแถวที่มีค่า+ป้ายของมัน (1307) ส่วนที่ทาดำหมดไม่มีไฟล์ bin (ตัวอ่านข้าม)
    index.json     รายการ section ตามลำดับอ่าน (id, kind, bbox, ขนาดภาพ, file = ชื่อไฟล์ หรือ null ถ้าทาดำหมด) ให้ตัวอ่านรู้ว่าไฟล์ไหนคือส่วนไหน
ให้ตัวอ่านอ่านทั้งสองแบบแล้วรวมตัวเลขที่เจอ (วัดบน 5 ใบ: รวมสองแบบได้ 19/30 ดีกว่าวิธีเดี่ยวทุกวิธีที่ได้ 16–17/30
ดู data/etc/experiments/section_preprocess/ และ docs) ส่วนที่ lab ตัดทิ้งเพราะว่างไม่ถูกสร้าง
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notebooks" / "preprocess_lab"))

import labkit as L  # noqa: E402
from fin_extractor.config import load_config  # noqa: E402
from fin_extractor.paths import DataPaths, dataset_of  # noqa: E402


def write_png(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError(f"encode ภาพไม่ได้: {path}")
    buf.tofile(str(path))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="โฟลเดอร์ชุดข้อมูล (ใช้เอาชื่อ dataset)")
    ap.add_argument("--only", nargs="*", help="เลขท้ายชื่อไฟล์ เช่น 1309")
    ap.add_argument("--drop-empty-rows", action="store_true",
                    help="แบบ bin: ลบแถวตารางที่ช่องตัวเลขว่าง (ฟอร์มพิมพ์รายการล่วงหน้า เช่น 50 ทวิ) ใช้คู่ --only ไม่ควรใช้กับใบกำกับที่ทุกแถวมีค่าอยู่แล้ว")
    ap.add_argument("--config", help="ไฟล์ config ทับ default.yaml (เปลี่ยน paths.data_dir ได้ ใช้ทดลองโดยไม่ทับผลจริง)")
    args = ap.parse_args()
    if args.config:
        L.PATHS = DataPaths(load_config(args.config))  # labkit ถามพาธจากตัวแปรนี้ตอนเรียก
    dataset = dataset_of(args.input)
    stems = sorted(p.stem for p in L.PATHS.section_features(dataset).glob("*.json"))
    if args.only:
        stems = [s for s in stems if any(s.endswith(o) for o in args.only)]
    total = 0
    clean_root = L.PATHS.section_clean(dataset)
    for stem in stems:
        old = clean_root / stem  # โครงเก่า clean/<ไฟล์>/sNN_<วิธี>.png
        if old.is_dir():
            shutil.rmtree(old)
        crops = [(sec, crop) for sec, crop in L.iter_section_crops(stem, dataset) if crop.size]
        bins = [(sec, L.binarize_inverted(crop)) for sec, crop in crops]
        if args.drop_empty_rows:
            dropped = L.drop_empty_rows_by_table(bins)
            infos = [info for _img, info in dropped]
            bins = [(sec, img) for (sec, _), (img, _info) in zip(bins, dropped)]
        else:
            infos = [{} for _ in bins]
        images = {"luma": [L.enhance.to_gray(c) for _, c in crops],
                  "flat": [L.flatten_background(L.gray_value_channel(c)) for _, c in crops],
                  "bin": [img for _, img in bins]}
        mb = 0.0
        for variant, imgs in images.items():
            out = L.PATHS.section_clean(dataset, stem, variant)
            if out.exists():
                shutil.rmtree(out)
            out.mkdir(parents=True)
            index = []
            for (sec, crop), img, info in zip(crops, imgs, infos):
                blank = variant == "bin" and info.get("blank")  # ทาดำหมดทั้งส่วน = ไม่มีไฟล์ ตัวอ่านข้าม (ยังอยู่ใน index ให้ตรงกับผัง)
                if not blank:
                    write_png(out / f"{sec['id']}.png", img)
                index.append({"id": sec["id"], "kind": sec["kind"], "bbox": sec["bbox"], "table": sec["table"],
                              **({"part": sec["part"], "parts": sec["parts"]} if sec.get("part") else {}),
                              "size": [int(crop.shape[1]), int(crop.shape[0])], "file": None if blank else f"{sec['id']}.png",
                              **({"rows": info} if variant == "bin" and info.get("applied") else {})})
            (out / "index.json").write_text(json.dumps({"source": stem, "page": 1, "variant": variant, "sections": index}, ensure_ascii=False, indent=1), encoding="utf-8")
            mb += sum(f.stat().st_size for f in out.glob("*.png")) / 1e6
        total += len(crops)
        print(f"{stem[-4:]}: {len(crops)} section × 3 วิธี · {mb:.1f} MB → {clean_root}/<วิธี>/{stem}", flush=True)
    print(f"รวม {total} section  จาก {len(stems)} ไฟล์")


if __name__ == "__main__":
    main()
