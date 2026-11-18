"""สร้างผังแบ่ง section ของทั้งชุดในครั้งเดียว (ไม่ต้องเปิด notebook ทีละเล่ม) ขั้นภาพล้วน ไม่ใช้ OCR/GPU

    python -X utf8 -m fin_extractor preprocess data/01_input/finance_doc_scan          # ต้องมีภาพที่ตรงแล้วก่อน
    python -X utf8 scripts/make_section_layouts.py data/01_input/finance_doc_scan
    python -X utf8 scripts/make_section_layouts.py data/01_input/finance_doc_scan --only 1309 1311 --table-mode cells
    python -X utf8 scripts/make_section_layouts.py data/01_input/finance_doc_scan --max-piece-frac 0      # ไม่ตัดส่วนสูงเป็นกลุ่มแถว

อินพุต: ภาพทั้งหน้าที่ preprocess แล้ว (``preprocess/<dataset>/<profile>/final/``)
วิธี: ย่อเป็นสำเนาทำงานด้านยาว ``labkit.WORK`` → เทา → Otsu (blur 3) → ``labkit.layout_sections`` (ค่าเดียวกับที่ notebook ใช้)
ผล: ``03_sections/<dataset>/<profile>/layout/<ไฟล์>.json`` เขียนเฉพาะคีย์ ``layout``, ``layout_params``, ``layout_sections``
    (คีย์อื่นที่ notebook เคยบันทึกไว้คงเดิม) และภาพตรวจด้วยตา ``<ไฟล์>_layout.png``
    (เขียว = ช่องตารางที่เก็บ, น้ำเงิน = ตารางทั้งก้อน/ท่อน, ส้ม = ข้อความนอกตาราง, แดงหนา = ว่าง)
    ส่วนที่สูงเกิน ``--max-piece-frac`` ถูกตัดเป็นกลุ่มแถว มี ``part``/``parts`` ใน ``layout_sections`` (ท่อนหลังไม่มีหัวคอลัมน์ซ้ำ)
ตามด้วย make_clean_sections.py ได้เลย (อ่านผังจากที่เดียวกัน)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fin_extractor.config import load_config  # noqa: E402
from fin_extractor.paths import DEFAULT_PROFILE, DataPaths, dataset_of  # noqa: E402
from fin_extractor.sections.build import build_layouts  # noqa: E402
from fin_extractor.sections.layout import MAX_PIECE_FRAC  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="โฟลเดอร์ชุดข้อมูล (ใช้เอาชื่อ dataset)")
    ap.add_argument("--only", nargs="*", help="เลขท้ายชื่อไฟล์ เช่น 1309")
    ap.add_argument("--profile", default=DEFAULT_PROFILE, help="ชุดตั้งค่า preprocess ที่เป็นต้นทางของภาพ")
    ap.add_argument("--table-mode", choices=("whole", "cells", "bands", "rows"), default="rows",
                    help="rows (ค่าตั้งต้น) = แถบเต็มความกว้างตัดที่ขอบตารางและเส้นแนวนอนในตารางด้วย · whole = ตารางทั้งก้อนเป็นส่วนเดียว · cells = แตกเป็นช่องตามเส้น · bands = แถบเต็มความกว้างตัดที่ขอบตารางเท่านั้น")
    ap.add_argument("--max-piece-frac", type=float, default=MAX_PIECE_FRAC,
                    help="ส่วนที่สูงกว่านี้ (สัดส่วนของความสูงหน้า) ถูกตัดเป็นกลุ่มแถวที่แถวว่างระหว่างบรรทัด · 0 = ไม่ตัด")
    ap.add_argument("--fill-aware", action="store_true", help="นับตัวอักษรบนพื้นสีเป็นหมึกและไม่ตัดผ่านแถบสี (ไม่ใช่ค่าตั้งต้น ดู docs/14)")
    ap.add_argument("--config", help="ไฟล์ config ทับ default.yaml (เปลี่ยน paths.data_dir ได้)")
    args = ap.parse_args()

    try:
        build_layouts(DataPaths(load_config(args.config)), dataset_of(args.input), args.profile, args.only, args.table_mode, args.max_piece_frac, args.fill_aware)
    except FileNotFoundError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
