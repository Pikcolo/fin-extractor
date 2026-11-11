"""เทียบผลสกัดของไฟล์เดียวรายฟิลด์ ข้ามทุกชุดผล (engine/โหมด) ที่อยู่ใน data/output/extract/

อ่านเฉพาะไฟล์ผลที่มีอยู่แล้ว ไม่เรียก OCR/LLM/GPU ผลเขียนที่ data/output/compare/<ไฟล์>/compare.md (และ compare.json)
ลบทิ้งทั้งโฟลเดอร์ได้เมื่อไม่ใช้

คอลัมน์ "อ้างอิง" มาจาก data/output/compare/<ไฟล์>/reference.json ถ้ามี (ไม่บังคับ) ไฟล์นั้นเป็นค่าที่คนอ่านจากภาพ ไม่ใช่เฉลยทางการ
ช่องที่มีอ้างอิงจะมีเครื่องหมาย ✓ (ตรง) / ✗ (ไม่ตรง) เทียบแบบตัดช่องว่าง ตัวเลขเทียบเป็นค่า

    .venv/Scripts/python scripts/compare_file.py 001312
    .venv/Scripts/python scripts/compare_file.py 001312 --runs paddle=ocr-rules_reextract_current typhoon=finance_doc_scan__vlm-rules
"""

from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTRACT = ROOT / "data" / "output" / "extract"
DEFAULT_RUNS = {
    "Paddle+rules": "ocr-rules_reextract_current",
    "Typhoon+rules": "finance_doc_scan__vlm-rules",
    "Typhoon+LLM": "finance_doc_scan__vlm-llm",
    "OXR+rules": "finance_doc_scan__oxr-rules",
}
FIELDS = [
    "document_type", "document_profile", "vendor.name", "vendor.branch", "vendor.tax_id", "vendor.address",
    "buyer.name", "buyer.branch", "buyer.tax_id", "buyer.address", "invoice_no", "book_no", "date",
    "items.count", "items[0].description", "items[0].quantity", "items[0].unit_price", "items[0].amount",
    "subtotal", "discount", "amount_before_vat", "vat_rate", "vat_amount", "total_amount",
]
SUMMARY = ["status", "missing_fields", "checks_failed"]


def get(result: dict, key: str):
    data = result["data"]
    if key == "items.count":
        return str(len(data["items"])) if data["items"] else None
    m = re.fullmatch(r"items\[(\d+)\]\.(\w+)", key)
    if m:
        i = int(m.group(1))
        return data["items"][i].get(m.group(2)) if i < len(data["items"]) else None
    node = data
    for part in key.split("."):
        node = node.get(part) if isinstance(node, dict) else None
    return node if node not in ("", [], {}) else None


def same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    try:
        return Decimal(str(a)) == Decimal(str(b))
    except InvalidOperation:
        return "".join(str(a).split()).casefold() == "".join(str(b).split()).casefold()


def cell(value, ref, has_ref: bool) -> str:
    text = "—" if value is None else str(value).replace("|", "/")
    text = text if len(text) <= 46 else text[:45] + "…"
    if not has_ref:
        return text
    return f"{text} {'✓' if same(value, ref) else '✗'}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", help="ส่วนท้ายของชื่อไฟล์ เช่น 001312")
    ap.add_argument("--runs", nargs="*", help="ชื่อ=โฟลเดอร์ใน data/output/extract/ (ค่าตั้งต้น: Paddle/Typhoon/OXR)")
    args = ap.parse_args()
    runs = dict(r.split("=", 1) for r in args.runs) if args.runs else DEFAULT_RUNS

    out_dir = ROOT / "data" / "output" / "compare" / args.stem.lstrip("0")
    ref_path = out_dir / "reference.json"
    ref = json.loads(ref_path.read_text(encoding="utf-8")) if ref_path.exists() else None

    results: dict[str, dict] = {}
    for label, folder in runs.items():
        hits = sorted((EXTRACT / folder / "results").glob(f"*{args.stem}*_p1.json"))
        if hits:
            results[label] = json.loads(hits[0].read_text(encoding="utf-8"))
        else:
            print(f"ไม่มีผล {args.stem} ใน {folder} (ข้าม {label})")
    if not results:
        raise SystemExit("ไม่พบผลของไฟล์นี้ในชุดผลใดเลย")

    cols = list(results)
    lines = [f"# เทียบผลสกัดไฟล์ {args.stem} รายฟิลด์", ""]
    if ref:
        lines += [f"> คอลัมน์อ้างอิง: {ref.get('_note', '')}", ""]
    lines += ["| ฟิลด์ | " + ("อ้างอิง | " if ref else "") + " | ".join(cols) + " |",
              "|---|" + ("---|" if ref else "") + "---|" * len(cols)]
    table: dict[str, dict] = {}
    doc_keys = sorted({f"document_fields.{k}" for r in results.values() for k in r["data"]["document_fields"]}
                      | {k for k in (ref or {}) if k.startswith("document_fields.")})  # ฟิลด์ประจำชนิดตามโปรไฟล์
    for key in [*FIELDS, *doc_keys]:
        vals = {c: get(results[c], key) for c in cols}
        expected = ref.get(key) if ref else None
        if all(v is None for v in vals.values()) and (not ref or key not in ref):
            continue
        table[key] = {"reference": expected, **vals}
        lines.append(f"| `{key}` | " + (f"{'—' if expected is None else str(expected)[:46]} | " if ref else "")
                     + " | ".join(cell(vals[c], expected, bool(ref) and key in ref) for c in cols) + " |")
    if ref:
        score = {c: sum(same(table[k][c], ref[k]) for k in ref if k in table) for c in cols}
        total = sum(1 for k in ref if k in table)
        lines += ["", "ตรงกับอ้างอิง (เฉพาะฟิลด์ที่มีในไฟล์อ้างอิง): " + ", ".join(f"{c} {score[c]}/{total}" for c in cols)]
    lines += ["", "| สรุป | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for key in SUMMARY:
        row = []
        for c in cols:
            r = results[c]
            v = {"status": r["status"], "missing_fields": ",".join(r["missing_fields"]) or "—",
                 "checks_failed": ",".join(x["name"] for x in r["checks"] if x["passed"] is False) or "—"}[key]
            row.append(v)
        lines.append(f"| {key} | " + " | ".join(row) + " |")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "compare.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "compare.json").write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved -> {out_dir}")


if __name__ == "__main__":
    main()
