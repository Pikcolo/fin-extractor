"""รันขั้นสกัด + validation ซ้ำจาก ``raw_text`` ที่เก็บไว้ (รัน extract ด้วย --keep-raw) โดยไม่เรียก OCR/VLM/GPU

ใช้ตรวจว่าแก้ขั้นสกัด (rules, โปรไฟล์ชนิดเอกสาร, validation) แล้วผลเปลี่ยนอย่างไร บนข้อความ OCR ชุดเดิมเป๊ะ
จึงเทียบก่อน/หลังได้โดยไม่ปนความต่างของ OCR และรันได้ในไม่กี่วินาทีบนเครื่องที่ไม่มี GPU

บรรทัดข้อความ = ``raw_text`` แยกด้วยบรรทัดใหม่ (โหมด ocr: raw_text คือข้อความของ engine เรียงตามบรรทัดอยู่แล้ว)
ผลที่ได้เป็น rules ล้วน (ไม่มี LLM)

    .venv/Scripts/python scripts/reextract_from_raw.py data/output/extract/_baseline_ocr_raw data/output/extract/_reextract
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fin_extractor.config import load_config  # noqa: E402
from fin_extractor.doctypes import load_core_labels, load_profiles  # noqa: E402
from fin_extractor.pipeline import FinanceOCRPipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="โฟลเดอร์ผล extract ที่มี results/*.json และ raw_text")
    ap.add_argument("out", help="โฟลเดอร์ปลายทาง")
    args = ap.parse_args()

    cfg = load_config()
    pipe = object.__new__(FinanceOCRPipeline)  # ไม่โหลด OCR/VLM — ใช้เฉพาะตรรกะสกัด/ตรวจ
    pipe.cfg, pipe.llm = cfg, None
    pipe.reader = SimpleNamespace(name="paddle(reextract)")  # ชื่อ engine ในผล
    pipe.doc_table = cfg.get("document_types") or {}
    pipe.profiles = load_profiles(pipe.doc_table)
    pipe.core_labels = load_core_labels(cfg.get("core_labels"))

    out = Path(args.out) / "results"
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted((Path(args.src) / "results").glob("*.json")):
        old = json.loads(f.read_text(encoding="utf-8"))
        raw = old.get("raw_text")
        if raw is None:
            sys.exit(f"{f.name}: ไม่มี raw_text (ต้องรัน extract ด้วย --keep-raw)")
        lines = [ln for ln in raw.split("\n") if ln.strip()]
        res = pipe.build_result(old["source_file"], old["page"], lines, raw, old.get("ocr_confidence"), [])
        (out / f.name).write_text(res.model_dump_json(indent=2), encoding="utf-8")
        print(f"{f.name}: {res.data.document_profile:24} {old['status']} -> {res.status}  missing={','.join(res.missing_fields) or '-'}")


if __name__ == "__main__":
    main()
