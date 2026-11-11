"""วัด OXR (SoMarkAI/OXR) กับชุดสังเคราะห์ ด้วยเกณฑ์เดียวกับ benchmark_cer.py จะได้เทียบกับ Paddle / Typhoon ตรง ๆ

เรียก OXR ผ่าน free hosted API (เครื่องนี้ VRAM 6GB รัน OXR เองไม่ได้ ดู docs/10-oxr.md)
**ส่งเฉพาะภาพสังเคราะห์ (ข้อมูลสมมติ) ห้ามชี้ไปที่สแกนจริง** เพราะ FAQ ของ OXR ระบุว่าข้อมูลบน free API ถูกใช้ปรับปรุงโมเดล

วัดสามอย่าง
- CER ระดับข้อความทั้งหน้า (เหมือน benchmark_cer.py) เทียบข้อความที่ render_page วาด
- field ที่ตัวอักษรตรงเป๊ะ: เลขภาษี, เลขที่ใบกำกับ, วันที่, ยอดสามตัว, ชื่อผู้ขาย (ค้นในข้อความที่ normalize แล้ว)
- จำนวนตัวอักษร CJK ที่ OXR คืนมา ทั้งที่ภาพไม่มี (ใช้บอกว่าโมเดลแต่งอักษร)

ผลดิบจาก API เก็บแคชที่ data/output/oxr/<tag>/ รันซ้ำไม่เรียก API อีก (ใส่ --refresh เพื่อเรียกใหม่)

    .venv/Scripts/python scripts/benchmark_oxr.py [--preprocess] [--refresh]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2
import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark_cer as bc  # noqa: E402
from fin_extractor.config import load_config  # noqa: E402
from fin_extractor.extraction import markdown_to_lines  # noqa: E402
from fin_extractor.preprocessing import preprocess  # noqa: E402

API = "https://free-oxr.somark.ai/api/v1/parse/sync"
CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")


def call_oxr(jpeg: bytes, name: str) -> dict:
    files = [("file", (name, jpeg, "image/jpeg")), ("output_formats", (None, "markdown")),
             ("output_formats", (None, "json")), ("api_key", (None, "dummy"))]
    for wait in (20, 40, 80, 160, None):  # free API จำกัดอัตราคำขอ (429) จึงรอแล้วลองใหม่
        r = httpx.post(API, files=files, timeout=240)
        if r.status_code != 429 or wait is None:
            break
        print(f"  429 รอ {wait}s", flush=True)
        time.sleep(wait)
    r.raise_for_status()
    time.sleep(8)  # เว้นระหว่างคำขอ
    return r.json()


def field_hits(truth_fields: dict, text: str) -> dict[str, bool]:
    norm = bc.normalize(text)
    hits = {}
    for k, v in truth_fields.items():
        want = f"{v:,.2f}" if isinstance(v, float) else str(v)
        hits[k] = bc.normalize(want) in norm
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="data/raw/synthetic")
    ap.add_argument("--preprocess", action="store_true", help="ผ่าน preprocess ของโปรเจกต์ก่อนส่ง (เทียบกับ Paddle/Typhoon ได้ตรง)")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    if "finance_doc_scan" in Path(args.samples).resolve().parts:  # สแกนจริงห้ามส่งขึ้น free API (ข้อมูลถูกใช้ปรับปรุงโมเดล)
        sys.exit("ปฏิเสธ: ห้ามส่งสแกนจริงขึ้น free API ใช้ได้เฉพาะภาพสังเคราะห์ (data/raw/synthetic)")

    tag = "preprocessed" if args.preprocess else "raw"
    cache = ROOT / "data/output/oxr" / tag
    cache.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    samples = Path(args.samples)
    truth = json.loads((samples / "truth.json").read_text(encoding="utf-8"))
    ref = bc.reference_text()

    rows = []
    for name, meta in truth["files"].items():
        cached = cache / (Path(name).stem + ".json")
        t0 = time.perf_counter()
        if cached.exists() and not args.refresh:
            resp = json.loads(cached.read_text(encoding="utf-8"))
        else:
            img = bc.read(samples / name)
            if args.preprocess:
                img = preprocess(img, cfg["preprocessing"]).image
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            resp = call_oxr(buf.tobytes(), name)
            cached.write_text(json.dumps(resp, ensure_ascii=False, indent=1), encoding="utf-8")
        md = resp["data"]["result"]["outputs"]["markdown"]
        hyp = "\n".join(markdown_to_lines(md))
        hits = field_hits(truth["fields"], hyp)
        row = {"file": name, "stamp": meta["stamp"], "scan_effects": meta["scan_effects"],
               "cer": round(bc.cer(ref, hyp), 4), "cjk_chars": len(CJK.findall(hyp)),
               "fields_ok": sum(hits.values()), "fields": hits,
               "api_ms": resp["data"]["metadata"]["processing_time_ms"],
               "seconds": round(time.perf_counter() - t0, 2)}
        rows.append(row)
        print(f"{name:42} CER {row['cer']:.4f}  CJK {row['cjk_chars']:3}  fields {row['fields_ok']}/7  api {row['api_ms']}ms", flush=True)

    n = len(truth["fields"])
    per_field = {k: sum(r["fields"][k] for r in rows) for k in truth["fields"]}
    summary = {"summary": bc.summarize(rows), "per_field_ok_of_%d" % len(rows): per_field,
               "files_with_cjk": sum(r["cjk_chars"] > 0 for r in rows),
               "mean_api_ms": round(sum(r["api_ms"] for r in rows) / len(rows)),
               "fields_total": f"{sum(r['fields_ok'] for r in rows)}/{n * len(rows)}"}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    out = ROOT / f"data/output/benchmarks/oxr_synthetic_{tag}.json"
    out.write_text(json.dumps({"reference_chars": len(bc.normalize(ref)), **summary, "files": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved ->", out)


if __name__ == "__main__":
    main()
