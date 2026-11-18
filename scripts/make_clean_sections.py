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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fin_extractor.config import load_config  # noqa: E402
from fin_extractor.paths import DataPaths, dataset_of  # noqa: E402
from fin_extractor.sections.build import build_images  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="โฟลเดอร์ชุดข้อมูล (ใช้เอาชื่อ dataset)")
    ap.add_argument("--only", nargs="*", help="เลขท้ายชื่อไฟล์ เช่น 1309")
    ap.add_argument("--drop-empty-rows", action="store_true",
                    help="แบบ bin: ลบแถวตารางที่ช่องตัวเลขว่าง (ฟอร์มพิมพ์รายการล่วงหน้า เช่น 50 ทวิ) ใช้คู่ --only ไม่ควรใช้กับใบกำกับที่ทุกแถวมีค่าอยู่แล้ว")
    ap.add_argument("--config", help="ไฟล์ config ทับ default.yaml (เปลี่ยน paths.data_dir ได้ ใช้ทดลองโดยไม่ทับผลจริง)")
    args = ap.parse_args()
    build_images(DataPaths(load_config(args.config)), dataset_of(args.input), args.only, args.drop_empty_rows)


if __name__ == "__main__":
    main()
