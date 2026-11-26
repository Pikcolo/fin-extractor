"""P0 spike ของ docs/draft-11: รัน OpenThai-SystemOne บน CPU กับผลที่เก็บไว้ของ 18 ใบ (ไม่ OCR ใหม่ ไม่แตะ src/)

วัด: เวลาโหลด · หน่วยความจำ (RSS) · เวลาต่อคำถาม · ความน่าจะเป็นของ abstain · คำตอบพลิกเมื่อสลับลำดับตัวเลือก
งาน: T1 จำแนกชนิดเอกสาร สองชุดตัวเลือก (ก) ชนิดตามที่พิมพ์ จากรายการของผู้ใช้ (ข) โปรไฟล์ใน document_types + generic
     T2 ตัวอย่าง: ชื่อผู้ขายเป็นชื่อบริษัท/บุคคลจริงไหม (noul)
state ที่ป้อน: ข้อความท่อนแรก (s00) ของ .md ตัดแท็ก HTML + ชื่อผู้ขาย/ผู้ซื้อที่สกัดได้ — ไม่ป้อนข้อความทั้งหน้า

    .venv/Scripts/python -X utf8 scripts/eval/systemone_spike.py [--model models/OpenThai-SystemOne] [--out ไฟล์.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fin_extractor.config import load_config  # noqa: E402
from fin_extractor.doctypes import load_profiles  # noqa: E402
from fin_extractor.extraction.rule_extractor import chunk_lines  # noqa: E402

OCR = ROOT / "data/04_ocr/finance_doc_scan/openthai/regions-luma"
EXTRACT = ROOT / "data/05_extract/finance_doc_scan/openthai/regions-luma/rules"

# เฉลย T1 (ก): ชนิดตามที่ผู้ใช้ระบุ (แชต 2026-11-26) ใช้ข้อความตามที่ให้มา ยังไม่ normalize
USER_TYPE = {
    "001306": "ใบแจ้งหนี้/ใบเสร็จ",
    **dict.fromkeys(["001307", "001311", "001321", "001323"], "หนังสือรับรองการหักภาษี ณ ที่จ่าย"),
    **dict.fromkeys(["001308", "001309", "001310", "001313", "001314"], "ใบเสร็จรับเงิน/ใบกำกับภาษี"),
    "001312": "ใบส่งของ/ใบเสร็จรับเงิน/ใบกำกับภาษี",
    "001315": "ใบเสร็จ (ENG ล้วน)",
    **dict.fromkeys(["001316", "001317", "001318"], "ใบเสร็จรับเงิน"),
    "001319": "ต้นฉบับใบส่งสินค้า/สำเนาใบกำกับภาษี",
    **dict.fromkeys(["001320", "001322"], "ใบสำคัญรับเงิน (ลายมือ)"),
}


def first_chunk(md: str) -> str:
    """ข้อความของท่อน s00 (region บนสุดของผัง) เป็นบรรทัดล้วน."""
    return "\n".join(ln for ln, c, s in chunk_lines(md) if c == "s00" and not s)


def orders(n: int) -> list[list[int]]:
    """ลำดับตัวเลือกที่ทดสอบ: เดิม · กลับด้าน · หมุนครึ่ง (กำหนดตายตัว ไม่สุ่ม รันซ้ำได้ผลเดิม)."""
    base = list(range(n))
    out = [base, base[::-1], base[n // 2:] + base[: n // 2]]
    return [o for i, o in enumerate(out) if o not in out[:i]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/OpenThai-SystemOne"))
    ap.add_argument("--out", default=str(ROOT / "data/etc/compare/systemone_spike.json"))
    args = ap.parse_args()

    from openthai_systemone import Choice, Noul, SystemOneClient

    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    t = time.perf_counter()
    client = SystemOneClient(args.model, device="cpu")
    load_s = time.perf_counter() - t
    rss_loaded = proc.memory_info().rss

    profiles = load_profiles(load_config()["document_types"])
    prof_opts = {p.name: p.title for p in profiles} | {"generic": "ใบกำกับภาษี / ใบเสร็จรับเงิน / ใบแจ้งหนี้ / ใบส่งของ ทั่วไป"}
    type_opts = {t: None for t in dict.fromkeys(USER_TYPE.values())}

    def ask(state, key, q_cls, instructions, opts):
        """ถามหนึ่งคำถามทุกลำดับตัวเลือก คืนผลแต่ละลำดับ + เวลา."""
        names = list(opts)
        runs = []
        for o in orders(len(names)):
            crit = {names[i]: opts[names[i]] for i in o}
            t0 = time.perf_counter()
            r = client.system_one(state, {key: q_cls(instructions=instructions, criteria=crit)})
            a = r.answers[key]
            runs.append({"order": [names[i] for i in o], "seconds": round(time.perf_counter() - t0, 3), "tokens": r.usage.input_tokens,
                         "choice": a.choice, "confidence": round(a.confidence, 4), "abstain": round(a.abstain, 4),
                         "probabilities": {k: round(v, 4) for k, v in a.probabilities.items()}})
        return runs

    docs = []
    peak = rss_loaded
    for md_path in sorted(glob.glob(str(OCR / "*" / "*_p1.md"))):
        n = Path(md_path).stem.split("_")[-2]
        res = json.loads(next(Path(EXTRACT).glob(f"*{n}/*_p1.json")).read_text(encoding="utf-8"))
        d = res["data"]
        state = {"ข้อความหัวเอกสาร (OCR)": first_chunk(Path(md_path).read_text(encoding="utf-8")),
                 "ชื่อผู้ขายที่สกัดได้": (d.get("vendor") or {}).get("name"),
                 "ชื่อผู้ซื้อที่สกัดได้": (d.get("buyer") or {}).get("name")}
        t1a = ask(state, "doc_type", Choice, "เอกสารนี้เป็นเอกสารชนิดใด ตามชื่อเอกสารที่พิมพ์บนหัวเอกสาร", type_opts)
        t1b = ask(state, "profile", Choice, "เอกสารนี้จัดอยู่ในกลุ่มใด", prof_opts)
        vname = (d.get("vendor") or {}).get("name")
        t2 = None
        if vname:
            t0 = time.perf_counter()
            r = client.system_one({"ค่าในช่องชื่อผู้ขาย": vname},
                                  {"is_name": Noul(instructions="ข้อความนี้เป็นชื่อบริษัทหรือชื่อบุคคลเท่านั้น ไม่มีข้อความอื่นปน ใช่หรือไม่")})
            t2 = {"value": vname, "p_yes": round(r.answers["is_name"].noul, 4), "seconds": round(time.perf_counter() - t0, 3)}
        peak = max(peak, proc.memory_info().rss)
        docs.append({"file": n, "user_type": USER_TYPE.get(n), "rules_document_type": d.get("document_type"),
                     "rules_profile": d.get("document_profile"), "state_chars": len(state["ข้อความหัวเอกสาร (OCR)"]),
                     "t1_type": t1a, "t1_profile": t1b, "t2_vendor_name": t2})
        print(f"{n}: type={t1a[0]['choice']} (abst {t1a[0]['abstain']}) · profile={t1b[0]['choice']} · "
              f"flip type={len({r['choice'] for r in t1a}) > 1} profile={len({r['choice'] for r in t1b}) > 1} · "
              f"{t1a[0]['seconds']}s", flush=True)

    secs = [r["seconds"] for d in docs for r in d["t1_type"] + d["t1_profile"]]
    summary = {
        "model": args.model, "device": "cpu", "torch_threads": __import__("torch").get_num_threads(),
        "load_seconds": round(load_s, 1), "rss_before_mb": rss0 // 2**20, "rss_after_load_mb": rss_loaded // 2**20, "rss_peak_mb": peak // 2**20,
        "questions": len(secs), "seconds_per_question": {"min": min(secs), "median": sorted(secs)[len(secs) // 2], "max": max(secs)},
        "t1_type_correct_original_order": sum(d["t1_type"][0]["choice"] == d["user_type"] for d in docs),
        "t1_type_flipped": sum(len({r["choice"] for r in d["t1_type"]}) > 1 for d in docs),
        "t1_profile_flipped": sum(len({r["choice"] for r in d["t1_profile"]}) > 1 for d in docs),
        "docs": len(docs),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "docs": docs}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("→", args.out)


if __name__ == "__main__":
    main()
