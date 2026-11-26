"""P3 ของ docs/draft-11: วัด decision model เทียบกฎเดิม จากผลที่ extract --decision on บันทึกไว้ (ไม่รันโมเดลใหม่)

T1 ชนิดเอกสาร (เฉลย: data/etc/groundtruth/<dataset>/doc_types.json ที่ผู้ใช้ให้มา) รายงาน 2 ระดับตามที่ผู้ใช้เลือก
    ตรงตัว   ชื่อชนิดหลังตัดหมายเหตุในวงเล็บ ต้องตรงทั้งชุด (rules ตอบได้แค่ 4 ค่าภาษาอังกฤษ จึงไม่มีคะแนนระดับนี้)
    กลุ่ม    กลุ่มที่มีผลกับงานบัญชี: withholding (50 ทวิ) · voucher (ใบสำคัญรับเงิน) · tax_invoice (ชื่อมีคำว่า ใบกำกับภาษี) · other
             กลุ่มของ rules มาจากโปรไฟล์ (50 ทวิ / ใบสำคัญรับเงิน) แล้วจาก document_type (tax_invoice → tax_invoice ที่เหลือ other)
T2 จาก Excel ตรวจผลที่ผู้ใช้กรอก (data/etc/compare/check_rule_hybrid_filled.xlsx หรือ check_rule_hybrid.xlsx ชีท rule)
    แถว = ฟิลด์ที่ขาด/ไม่ถูก (ระบบใส่ให้) + ที่ผู้ใช้เพิ่ม · ฟิลด์ถือว่า "ผิด" เมื่อ ผลตรวจ ไม่ว่างและไม่ใช่ "ถูก" หรือ ค่าที่ถูกต้อง ต่างจากค่าที่ระบบได้
    name_only: ชื่อที่ผิดตามชีท ↔ โมเดลตอบ no ไหม · ชื่อที่ไม่อยู่ในชีทนับเป็น "ไม่ทราบ" ไม่เดาว่าถูก
    missing_hint: ค่าที่โมเดลเสนอ ↔ ค่าที่ถูกต้องที่ผู้ใช้กรอก

    .venv/Scripts/python -X utf8 scripts/eval/decision_p3.py [--extract <โฟลเดอร์ผล rules>] [--md]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXTRACT = ROOT / "data/05_extract/finance_doc_scan/openthai/regions-luma/rules"
GT = ROOT / "data/etc/groundtruth/finance_doc_scan"


def normalize(t: str | None) -> str | None:
    """ตัดหมายเหตุในวงเล็บ · 'ใบเสร็จ' เปล่า ๆ = 'ใบเสร็จรับเงิน' (บันทึกไว้ใน doc_types.json → normalization)."""
    if t is None:
        return None
    t = re.sub(r"\s*\([^)]*\)", "", t).strip()
    return "ใบเสร็จรับเงิน" if t == "ใบเสร็จ" else t


def group_of_type(t: str | None) -> str | None:
    if t is None:
        return None
    if "หนังสือรับรองการหักภาษี" in t:
        return "withholding"
    if "ใบสำคัญรับเงิน" in t:
        return "voucher"
    return "tax_invoice" if "ใบกำกับภาษี" in t else "other"


def group_of_rules(doc_type: str | None, profile: str | None) -> str | None:
    if profile == "withholding_certificate":
        return "withholding"
    if profile == "payment_voucher":
        return "voucher"
    if doc_type is None:
        return None
    return "tax_invoice" if doc_type == "tax_invoice" else "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", type=Path, default=EXTRACT)
    args = ap.parse_args()
    truth = {k: normalize(v) for k, v in json.loads((GT / "doc_types.json").read_text(encoding="utf-8"))["as_given"].items()}

    rows = []
    for p in sorted(args.extract.rglob("*_p1.json")):
        n = p.stem.split("_")[-2]
        r = json.loads(p.read_text(encoding="utf-8"))
        decs = {d["task"]: d for d in r.get("decisions", []) if d["task"] in ("doc_type", "profile")}
        if "doc_type" not in decs or n not in truth:
            continue
        dt = decs["doc_type"]
        model = normalize(dt["answer"])
        rows.append({"file": n, "truth": truth[n], "model": model, "agreed": dt["agreed"],
                     "p_top": max(dt["probabilities"].values()), "abstain": dt.get("abstain"),
                     "rules_type": r["data"].get("document_type"), "rules_profile": r["data"].get("document_profile"),
                     "model_profile": decs.get("profile", {}).get("answer"),
                     "exact_ok": model == truth[n],
                     "group_truth": group_of_type(truth[n]), "group_model": group_of_type(model),
                     "group_rules": group_of_rules(r["data"].get("document_type"), r["data"].get("document_profile"))})
    if not rows:
        raise SystemExit(f"ไม่มีผลที่มี decisions ใน {args.extract} — รัน extract --decision on ก่อน")

    n = len(rows)
    exact = sum(x["exact_ok"] for x in rows)
    gm = sum(x["group_model"] == x["group_truth"] for x in rows)
    gr = sum(x["group_rules"] == x["group_truth"] for x in rows)
    wrong = [x for x in rows if not x["exact_ok"]]
    print(f"T1 ชนิดเอกสาร · {n} ใบ")
    print(f"  ตรงตัว (โมเดล)          {exact}/{n}")
    print(f"  ระดับกลุ่ม  โมเดล {gm}/{n} · rules {gr}/{n}")
    print(f"  โมเดลถูก rules ผิด (กลุ่ม): {[x['file'] for x in rows if x['group_model'] == x['group_truth'] != x['group_rules']]}")
    print(f"  rules ถูก โมเดลผิด (กลุ่ม): {[x['file'] for x in rows if x['group_rules'] == x['group_truth'] != x['group_model']]}")
    dis = [x for x in rows if not x["agreed"]]
    print(f"  ลำดับตัวเลือกให้คำตอบต่างกัน {len(dis)} ใบ: ในนั้นผิดตรงตัว {sum(not x['exact_ok'] for x in dis)} · "
          f"ใบที่ผิดตรงตัวทั้งหมด {len(wrong)} ใบ ลำดับไม่ตรงกัน {sum(not x['agreed'] for x in wrong)}")
    print(f"  abstain สูงสุด {max(x['abstain'] or 0 for x in rows):.4f}")
    print(f"  โปรไฟล์ โมเดลตรงกับ rules {sum(x['model_profile'] == x['rules_profile'] for x in rows)}/{n}")
    print("\n| ไฟล์ | เฉลย | โมเดล | ตรงตัว | ลำดับตรงกัน | p สูงสุด | กลุ่ม เฉลย / โมเดล / rules |\n|---|---|---|---|---|---|---|")
    for x in rows:
        print(f"| {x['file']} | {x['truth']} | {x['model']} | {'✓' if x['exact_ok'] else '✗'} | {'✓' if x['agreed'] else '✗'} | "
              f"{x['p_top']:.2f} | {x['group_truth']} / {x['group_model']} / {x['group_rules']} |")

    xl = next((p for p in (ROOT / "data/etc/compare/check_rule_hybrid_filled.xlsx", ROOT / "data/etc/compare/check_rule_hybrid.xlsx")
               if p.exists()), None)
    if xl is None:
        print("\nT2 ข้าม: ยังไม่มีไฟล์ตรวจผล")
        return
    from openpyxl import load_workbook

    ws = load_workbook(xl, data_only=True)["rule"]
    col = {c.value: c.column - 1 for c in ws[1] if c.value}
    marks = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row[col["ไฟล์"]]:
            continue
        verdict, fix, got = row[col["ผลตรวจ"]], row[col["ค่าที่ถูกต้อง"]], row[col["ค่าที่ระบบได้"]]
        if verdict is None and fix is None:
            continue  # ยังไม่ได้ตรวจแถวนี้
        wrong = (verdict is not None and str(verdict).strip() != "ถูก") or (fix is not None and str(fix).strip() != str(got or "").strip())
        marks[(str(row[col["ไฟล์"]]), row[col["ฟิลด์ (key)"]])] = (wrong, fix)
    print(f"\nT2 จาก {xl.name} · แถวที่ตรวจแล้ว {len(marks)}")
    if not marks:
        print("  ยังไม่มีแถวที่กรอก ผลตรวจ / ค่าที่ถูกต้อง")
        return
    flagged = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "unknown": 0}
    hints = []
    for p in sorted(args.extract.rglob("*_p1.json")):
        fno = p.stem.split("_")[-2]
        for d in json.loads(p.read_text(encoding="utf-8")).get("decisions", []):
            m = marks.get((fno, d["key"]))
            if d["task"] == "name_only":
                if m is None:
                    flagged["unknown"] += 1
                    continue
                bad, no = m[0], d["answer"] == "no"
                flagged["tp" if bad and no else "fn" if bad else "fp" if no else "tn"] += 1
            elif d["task"] == "missing_hint" and m and m[1] is not None:
                hints.append((fno, d["key"], d["answer"], str(m[1]), d["agreed"]))
    print(f"  name_only: ชื่อที่ผิดจริง {flagged['tp'] + flagged['fn']} (โมเดลตอบ no {flagged['tp']}) · "
          f"ชื่อที่ถูก {flagged['fp'] + flagged['tn']} (โมเดลตอบ no {flagged['fp']}) · ไม่อยู่ในชีท {flagged['unknown']}")
    for fno, key, ans, fix, agreed in hints:
        print(f"  missing_hint {fno} {key}: เสนอ {ans} · ค่าที่ถูก {fix} · {'ตรง' if ans.replace(',', '') == fix.replace(',', '') else 'ไม่ตรง'}"
              f"{'' if agreed else ' (ลำดับตัวเลือกให้คำตอบต่างกัน)'}")

if __name__ == "__main__":
    main()
