"""อัตราที่คนตรวจแก้ค่าของแต่ละระดับหลักฐาน จาก edits.json ที่หน้าตรวจ/แก้บันทึกไว้ (ไม่ต้องกรอกเฉลยแยก)

ทุกเอกสารผ่านคนตรวจอยู่แล้ว (HITL) ทุกครั้งที่กดบันทึก ช่องที่อยู่ใน ``edits`` = ระบบได้ค่าผิด/ขาด ช่องที่ไม่อยู่ = คนตรวจแล้วคงไว้
นับ ถูกแก้ / ทั้งหมด แยกตาม ระดับ · ระดับ × กลุ่มฟิลด์ · check ที่ยืนยันช่องนั้น (ระดับ verified)

ข้อจำกัด (บอกในรายงานด้วย)
- ช่องที่ไม่ถูกแก้อาจเป็นเพราะคนตรวจไม่ได้ดูละเอียด อัตรานี้จึงเป็นขอบล่างของอัตราผิด
- ไม่มีการแก้เลยใน n ช่อง ไม่ได้แปลว่าผิด 0%: ขอบบนที่ความเชื่อมั่น 95% ≈ 3/n (rule of three,
  Hanley & Lippman-Hand 1983, JAMA 249:1743) มีการแก้ใช้ช่วง Wilson 95% (Wilson 1927, JASA 22:209)
- นับเฉพาะ edits.json ที่มี ``review`` (บันทึกจากหน้าเว็บรุ่นที่เก็บระดับก่อนแก้) · รายการสินค้านับเป็นกลุ่มเดียว ("items")

    .venv/Scripts/python -X utf8 scripts/eval/edit_rates.py [--root data/etc/demo]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from fin_extractor.validation.evidence import NOT_CORROBORATING

ROOT = Path(__file__).resolve().parents[2]
Z95 = 1.959964  # ควอนไทล์ 0.975 ของการแจกแจงปกติ (ช่วงสองข้าง 95%)


def group_of(key: str) -> str:
    if key.startswith("items["):
        return "items"
    if key.startswith("document_fields."):
        return "document_fields"
    if key.startswith(("vendor.", "buyer.")):
        return key.split(".", 1)[1] if key.endswith(("tax_id", "name")) else "party_other"
    return {"date": "date", "invoice_no": "invoice_no", "book_no": "invoice_no"}.get(key, "amount")


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + Z95 ** 2 / n
    mid = (p + Z95 ** 2 / (2 * n)) / den
    half = Z95 * math.sqrt(p * (1 - p) / n + Z95 ** 2 / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def rate(k: int, n: int) -> str:
    if n == 0:
        return "—"
    if k == 0:
        return f"0/{n} (ผิดไม่เกิน ~{min(1, 3 / n):.0%} ที่ 95%)"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.0%} (95%: {lo:.0%}–{hi:.0%})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "data/etc/demo")
    args = ap.parse_args()

    by_level, by_group, by_check = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    docs = skipped = 0
    for p in sorted(args.root.rglob("edits.json")):
        e = json.loads(p.read_text(encoding="utf-8"))
        if "review" not in e:
            skipped += 1
            continue
        docs += 1
        edited = {x["key"] for x in e["edits"]}
        passed = defaultdict(list)
        for c in e.get("checks", []):
            if c["passed"] is True and c["name"] not in NOT_CORROBORATING:
                for f in c["fields"]:
                    passed[f].append(c["name"])
        for key, rv in e["review"].items():
            if rv["level"] == "not_applicable":
                continue
            hit = key in edited or (key.startswith("items[") and "items" in edited)
            for bucket, k in ((by_level, rv["level"]), (by_group, (rv["level"], group_of(key)))):
                bucket[k][0] += hit
                bucket[k][1] += 1
            if rv["level"] == "verified":
                for name in passed.get(key, []):
                    by_check[name][0] += hit
                    by_check[name][1] += 1

    print(f"เอกสารที่ตรวจแล้ว {docs} ใบ" + (f" (ข้าม {skipped} ไฟล์ที่ไม่มีระดับก่อนแก้)" if skipped else ""))
    if not docs:
        print("ยังไม่มีข้อมูล: ตรวจ/แก้ในหน้าเว็บเดโมแล้วกด 'บันทึก' (เขียน edits.json ในโฟลเดอร์งาน)")
        return
    print("\nถูกแก้ / ทั้งหมด ต่อระดับ")
    for lv in ("conflict", "missing", "unverified", "verified"):
        print(f"  {lv:11s} {rate(*by_level[lv])}")
    print("\nต่อระดับ × กลุ่มฟิลด์")
    for (lv, g), (k, n) in sorted(by_group.items()):
        print(f"  {lv:11s} {g:16s} {rate(k, n)}")
    if by_check:
        print("\nช่อง verified ต่อ check ที่ยืนยัน (ช่องหนึ่งนับได้หลาย check)")
        for name, (k, n) in sorted(by_check.items()):
            print(f"  {name:22s} {rate(k, n)}")
    print("\nหมายเหตุ: ช่องที่ไม่ถูกแก้อาจเป็นเพราะคนตรวจไม่ได้ดูละเอียด อัตรานี้จึงเป็นขอบล่างของอัตราผิด")


if __name__ == "__main__":
    main()
