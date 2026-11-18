"""รัน OCR ทีละภาพย่อย (section) ของทุกไฟล์ที่แบ่งไว้ด้วย notebooks/preprocess_lab แล้วเก็บเป็น raw_data

    python -X utf8 scripts/run_sections.py data/01_input/finance_doc_scan                     # --engine openthai --variant luma (ค่าตั้งต้น)
    python -X utf8 scripts/run_sections.py data/01_input/finance_doc_scan --engine oxr
    python -X utf8 scripts/run_sections.py data/01_input/finance_doc_scan --variant flat --only 1307     # อ่านภาพ 03_sections/…/images/flat/
    python -X utf8 scripts/run_sections.py data/01_input/finance_doc_scan --variant color               # ตัดจากภาพสีใน memory (ไม่ผ่าน luma/flat)
    python -X utf8 scripts/run_sections.py data/01_input/finance_doc_scan --only 1309 1311 --redo

อินพุต: ผังแบ่ง ``03_sections/<dataset>/<profile>/layout/<ไฟล์>.json`` (make_section_layouts.py) + ภาพ ``--variant``
    luma (ค่าตั้งต้น) / flat / bin (ตัวขาวบนดำ) = ไฟล์ภาพย่อย ``03_sections/…/images/<แบบ>/<ไฟล์>/sNN.png`` (make_clean_sections.py) ตรวจก่อนว่าตรงกับผังปัจจุบัน ไม่งั้นหยุด
    color                    = ตัดจากภาพ final สีในหน่วยความจำตามผัง ไม่มีไฟล์ภาพย่อย ไม่ผ่าน luma/flat
ตัวอ่าน (เลือกด้วย ``--engine``) ตั้งค่าที่ section ชื่อเดียวกันใน configs/default.yaml
    oxr      ``vlm/oxr_local.py``    (``oxr_local:``) โหลดตรง ไม่ใช้ Docker
    openthai ``vlm/openthai_ocr.py`` (``openthai:``)
ผล: ``data/04_ocr/<dataset>/<engine>/<แบบ>/<ไฟล์>/<ไฟล์>_p1.sections.json`` = รายการ section ตามลำดับอ่าน (color = raw_data)
    {id, kind, bbox, text, truncated, seconds} ส่วนที่ lab ตัดทิ้ง (ว่าง) ไม่ถูกอ่าน เก็บเป็น ``skipped``
นอกจากนี้เขียน ``<ไฟล์>_p1.md`` ไว้ข้างกัน = ข้อความของทุกส่วนต่อกันตามลำดับอ่าน (ป้ายแต่ละส่วนเป็นคอมเมนต์ HTML ซึ่ง rules ตัดทิ้งตอนอ่าน)
เป็นรูปแบบเดียวกับข้อความดิบของโหมด VLM จึงใช้กับ ``extract --reuse-raw`` ได้ ``--markdown-only`` สร้าง .md จากผลที่มีอยู่โดยไม่โหลดโมเดล (ไม่ใช้ GPU)
รันต่อจากที่ค้างได้ (ข้ามไฟล์ที่มีผลแล้ว ใส่ --redo เพื่อทำใหม่) บันทึกผลทีละไฟล์

OXR ไม่รู้ภาษาไทย: ช่องที่มีไทยจะได้ข้อความมั่ว/อักษรจีน และอาจวนซ้ำจนชนเพดาน token (ช้า) ผลนี้เป็นข้อมูลดิบ ยังไม่ได้คัดว่าช่องไหนเชื่อได้
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fin_extractor import sections as S  # noqa: E402
from fin_extractor.config import load_config  # noqa: E402
from fin_extractor.paths import DataPaths, dataset_of  # noqa: E402
from fin_extractor.sections.read import (  # noqa: E402,F401  (tests และสคริปต์อื่นเรียกผ่านชื่อนี้)
    ENGINES, clean_index, fingerprint, image_locator, kept_layout, markdown_blocks, markdown_of, read_clean_image, read_dataset, chunks_of,
    script_mix, write_markdown)

VARIANTS = ("color", *S.VARIANTS)  # color = ตัดจากภาพ final ในหน่วยความจำ ที่เหลือคือภาพที่ make_clean_sections เขียนไว้


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="โฟลเดอร์ชุดข้อมูลเดียวกับที่ใช้สร้าง notebook (ใช้เอาชื่อ dataset)")
    ap.add_argument("--only", nargs="*", help="เลขท้ายชื่อไฟล์ เช่น 1309 (ไม่ใส่ = ทุกไฟล์ที่มีผังแบ่ง)")
    ap.add_argument("--redo", action="store_true", help="ทำใหม่แม้มีผลแล้ว")
    ap.add_argument("--markdown-only", action="store_true", help="สร้าง .md จากผลที่มีอยู่ ไม่โหลดโมเดล ไม่อ่านใหม่ (ไม่ใช้ GPU)")
    ap.add_argument("--config", help="ไฟล์ config ทับ default.yaml (เปลี่ยน paths.data_dir ได้ ใช้ทดลองโดยไม่ทับผลจริง)")
    ap.add_argument("--engine", choices=ENGINES, default="openthai", help="ตัวอ่านที่ใช้ (ผลลงใต้ 04_ocr/<dataset>/<engine>/)")
    ap.add_argument("--variant", choices=VARIANTS, default="luma",
                    help="ภาพที่ส่งเข้าตัวอ่าน: luma (ค่าตั้งต้น) / flat / bin (ตัวขาวบนดำ) = ไฟล์ใน 03_sections/…/images/<วิธี>/ (สร้างด้วย make_clean_sections.py หรือ fin_extractor sections) · "
                         "color = ตัดจากภาพ final สีในหน่วยความจำ (ไม่ผ่านการเตรียมรายส่วน) ผลแต่ละแบบแยกโฟลเดอร์ ไม่ทับกัน")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = DataPaths(cfg)
    dataset = dataset_of(args.input)
    layout_dir = paths.section_layout(dataset)
    stems = sorted(p.stem for p in layout_dir.glob("*.json"))
    if args.only:
        stems = [s for s in stems if any(s.endswith(o) for o in args.only)]
    if not stems:
        sys.exit(f"ไม่พบผังแบ่งใน {layout_dir} — รัน scripts/make_section_layouts.py ก่อน")
    t0 = time.perf_counter()
    summary = read_dataset(paths, cfg, dataset, stems, args.engine, args.variant, redo=args.redo, markdown_only=args.markdown_only)
    if summary:
        print("\nสรุป:")
        for i in summary:
            print(i)
        print(f"ไฟล์ที่รันรอบนี้ {len(summary)} · เวลารวม {(time.perf_counter() - t0) / 60:.1f} นาที")


if __name__ == "__main__":
    main()
