"""P3 ของ docs/draft-11: วัด decision model เทียบกฎเดิม จากผลที่ extract --decision on บันทึกไว้ (ไม่รันโมเดลใหม่)

T1 ชนิดเอกสาร (เฉลย: data/etc/groundtruth/<dataset>/doc_types.json ที่ผู้ใช้ให้มา) รายงาน 2 ระดับตามที่ผู้ใช้เลือก
    ตรงตัว   ชื่อชนิดหลังตัดหมายเหตุในวงเล็บ ต้องตรงทั้งชุด (rules ตอบได้แค่ 4 ค่าภาษาอังกฤษ จึงไม่มีคะแนนระดับนี้)
    กลุ่ม    กลุ่มที่มีผลกับงานบัญชี: withholding (50 ทวิ) · voucher (ใบสำคัญรับเงิน) · tax_invoice (ชื่อมีคำว่า ใบกำกับภาษี) · other
             กลุ่มของ rules มาจากโปรไฟล์ (50 ทวิ / ใบสำคัญรับเงิน) แล้วจาก document_type (tax_invoice → tax_invoice ที่เหลือ other)
T2 ไม่ใช้เฉลยที่คนกรอก (ผู้ใช้ไม่กรอก): เทียบผลที่เติมค่าแล้ว (extract --decision on, decision.fill_confirmed) กับผลก่อนเติม (--before)
    นับต่อใบ: ช่องที่ขาด ก่อน → หลัง · ข้อเสนอ · ข้อเสนอที่ check ยืนยัน (ชื่อ check) · ที่เติมจริง · สถานะ ก่อน → หลัง
    และยืนยันว่าค่าที่มีอยู่ก่อนไม่เปลี่ยน (เติมได้แค่ช่องที่ว่าง) · ด่านของการเติม = check ในเอกสารเอง จึงวัดได้โดยไม่มีเฉลย
    สิ่งที่วัดไม่ได้โดยไม่มีเฉลย: name_only ว่าจับชื่อผิดได้กี่ชื่อ (รายงานแค่จำนวนที่ตอบ no) และข้อเสนอที่ไม่มี check ยืนยันถูกหรือไม่

    .venv/Scripts/python -X utf8 scripts/eval/decision_p3.py [--extract <ผลที่เติมแล้ว>] --before <ผล extract --decision off>
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
    ap.add_argument("--before", type=Path, help="ผล extract --decision off ของชุดเดียวกัน (สำหรับ T2)")
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

    if args.before is None:
        print("\nT2 ข้าม: ใส่ --before <โฟลเดอร์ผล extract --decision off> เพื่อเทียบก่อน/หลังเติม")
        return
    flat = lambda d, pre="": {k2: v2 for k, v in d.items() for k2, v2 in (flat(v, f"{pre}{k}.").items() if isinstance(v, dict) else [(pre + k, v)])}
    tot = {"missing_before": 0, "missing_after": 0, "hints": 0, "confirmed": 0, "filled": 0, "changed": 0, "name_no": 0, "names": 0}
    print("\nT2 เติมค่าที่ check ยืนยัน (ไม่ใช้เฉลย)\n| ไฟล์ | ขาด ก่อน → หลัง | ข้อเสนอ | check ยืนยัน | เติม | สถานะ ก่อน → หลัง |\n|---|---|---|---|---|---|")
    for p in sorted(args.extract.rglob("*_p1.json")):
        q = next(args.before.rglob(p.name), None)
        if q is None:
            continue
        a, b = json.loads(p.read_text(encoding="utf-8")), json.loads(q.read_text(encoding="utf-8"))
        miss = lambda r: [k for k, v in r["review"].items() if v["level"] == "missing"]
        decs = a.get("decisions", [])
        hints = [d for d in decs if d["task"] == "missing_hint"]
        names = [d for d in decs if d["task"] == "name_only"]
        fa, fb = flat(a["data"]), flat(b["data"])
        changed = [k for k, v in fb.items() if v not in (None, "", [], {}) and fa.get(k) != v]
        tot["missing_before"] += len(miss(b)); tot["missing_after"] += len(miss(a)); tot["hints"] += len(hints)
        tot["confirmed"] += sum(bool(d.get("confirmed_by")) for d in hints); tot["filled"] += sum(bool(d.get("filled")) for d in hints)
        tot["changed"] += len(changed); tot["names"] += len(names); tot["name_no"] += sum(d["answer"] == "no" for d in names)
        print(f"| {p.stem.split('_')[-2]} | {len(miss(b))} → {len(miss(a))} | "
              f"{', '.join(f'{d['key']}={d['answer']}' for d in hints) or '—'} | "
              f"{', '.join(f'{d['key']}: {'+'.join(d['confirmed_by'])}' for d in hints if d.get('confirmed_by')) or '—'} | "
              f"{', '.join(d['key'] for d in hints if d.get('filled')) or '—'} | {b['status']} → {a['status']} |"
              + (f" ค่าเดิมเปลี่ยน: {changed}" if changed else ""))
    print(f"\nรวม: ช่องที่ขาด {tot['missing_before']} → {tot['missing_after']} · ข้อเสนอ {tot['hints']} · check ยืนยัน {tot['confirmed']} · "
          f"เติม {tot['filled']} · ค่าเดิมที่เปลี่ยน {tot['changed']} (ต้องเป็น 0)")
    print(f"name_only: ถาม {tot['names']} ชื่อ ตอบว่าไม่ใช่ชื่อล้วน {tot['name_no']} (ไม่มีเฉลย จึงบอกไม่ได้ว่าจับถูกกี่ชื่อ)")


if __name__ == "__main__":
    main()
