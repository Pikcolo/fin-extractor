"""เทียบผล ``extract`` สองโหมดรายใบ (ocr = Paddle + rules, vlm = Typhoon OCR + LLM + rules)

อ่าน JSON ของสองโฟลเดอร์ที่ชื่อไฟล์ตรงกัน แล้วเขียนตารางเทียบลง ``--out`` (Markdown)
ไม่มีเฉลย จึงเทียบเฉพาะสิ่งที่ตรวจเองได้: status, field สำคัญที่ขาด, check ที่ผ่าน/ไม่ผ่าน และค่าที่สองโหมดอ่านไม่ตรงกัน

    .venv/Scripts/python scripts/compare_modes.py [--a <results ของโหมด ocr>] [--b <results ของโหมด vlm>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FIELDS = ("vendor.name", "vendor.tax_id", "buyer.tax_id", "invoice_no", "date", "subtotal", "total_amount")


def get(d: dict, dotted: str):
    for part in dotted.split("."):
        d = (d or {}).get(part)
    return d


def summary(r: dict) -> dict:
    checks = r["checks"]
    return {
        "status": r["status"],
        "missing": r["missing_fields"],
        "pass": sum(c["passed"] is True for c in checks),
        "fail": sum(c["passed"] is False for c in checks),
        "items": len(r["data"]["items"]),
        "data": r["data"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="data/output/extract/finance_doc_scan__ocr-rules/results", help="โหมด ocr")
    ap.add_argument("--b", default="data/output/extract/finance_doc_scan__vlm-llm/results", help="โหมด vlm")
    ap.add_argument("--out", default="data/output/benchmarks/compare_modes.md")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # คอนโซล Windows เป็น cp1252

    a_dir, b_dir = Path(args.a), Path(args.b)
    names = sorted(p.name for p in a_dir.glob("*.json") if (b_dir / p.name).exists())
    rows, diffs = [], []
    counts = {"a": {}, "b": {}}
    for name in names:
        a = summary(json.loads((a_dir / name).read_text(encoding="utf-8")))
        b = summary(json.loads((b_dir / name).read_text(encoding="utf-8")))
        for k, s in (("a", a), ("b", b)):
            counts[k][s["status"]] = counts[k].get(s["status"], 0) + 1
        short = name.removeprefix("BRWAC50DE3689DA_").removesuffix("_p1.json")
        rows.append(
            f"| {short} | {a['status']} | {a['pass']}✓ {a['fail']}✗ | {','.join(a['missing']) or '-'} "
            f"| {b['status']} | {b['pass']}✓ {b['fail']}✗ | {','.join(b['missing']) or '-'} |"
        )
        for f in FIELDS:
            va, vb = get(a["data"], f), get(b["data"], f)
            if va != vb:
                diffs.append(f"| {short} | `{f}` | {va} | {vb} |")

    out = [
        "# เทียบสองโหมด `extract` บน `data/raw/finance_doc_scan`",
        "",
        f"A = `{args.a}` (--mode ocr: Paddle + rules) · B = `{args.b}` (--mode vlm: Typhoon OCR + LLM + rules) · {len(names)} ใบ ไม่มีเฉลย",
        "",
        f"สถานะ A: {counts['a']} · สถานะ B: {counts['b']}",
        "",
        "| ใบ | A status | A checks | A ขาด | B status | B checks | B ขาด |",
        "|---|---|---|---|---|---|---|",
        *rows,
        "",
        "## ค่าที่สองโหมดสกัดได้ไม่ตรงกัน",
        "",
        "ไม่ได้บอกว่าฝั่งไหนถูก (ไม่มีเฉลย) ใช้ตรวจด้วยตากับภาพ",
        "",
        "| ใบ | field | A (ocr) | B (vlm) |",
        "|---|---|---|---|",
        *diffs,
        "",
    ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(out), encoding="utf-8")
    print(f"saved -> {args.out}  ({len(names)} ใบ, ต่างกัน {len(diffs)} ค่า)")


if __name__ == "__main__":
    main()
