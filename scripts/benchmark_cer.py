"""วัด Character Error Rate (CER) กับชุดสังเคราะห์ที่รู้ข้อความจริง

ข้อความอ้างอิงไม่ได้พิมพ์ซ้ำด้วยมือ แต่ดึงจาก ``make_synthetic_samples.render_page`` โดยดักทุกคำสั่งวาดข้อความ
(ใบทุกใบเนื้อหาเดียวกัน) เรียงตาม (y, x) เหมือนที่ OCR จัดบรรทัด จึงไม่ต้องสร้างเฉลยเพิ่มและภาพเดิมไม่ถูกเปลี่ยน

    CER = ระยะแก้ไข (Levenshtein, ระดับอักขระ) ระหว่างข้อความที่อ่านได้กับข้อความอ้างอิง / ความยาวข้อความอ้างอิง

ทั้งสองฝั่งผ่าน normalize เดียวกัน: NFC, ``ํา`` → ``ำ``, ตัดช่องว่างทุกชนิดและ ``|`` (ตัวคั่นเซลล์ตารางของ VLM)
ฝั่ง VLM แปลง Markdown/HTML เป็นบรรทัดก่อน เพื่อไม่ให้แท็กนับเป็นข้อผิดพลาด

ข้อควรระวังเวลาอ่านผล
- CER ไวต่อ *ลำดับ* บรรทัด: ถ้า engine อ่านบรรทัดสลับที่ ตัวอักษรถูกก็ยังนับผิด
- ใบที่มีตรายาง (``stamp``) ข้อความตรา ("ชำระเงินแล้ว", "PAID ...") ไม่อยู่ในข้อความอ้างอิง ถ้าอ่านได้จะนับเป็นตัวอักษรเกิน
  จึงรายงานแยกจากใบไม่มีตรา
- เป็นภาพสังเคราะห์ที่ฟอนต์สะอาด ไม่ใช่ CER ของเอกสารจริง สแกนจริงไม่มีข้อความอ้างอิงจึงวัด CER ไม่ได้

    .venv/Scripts/python scripts/benchmark_cer.py [--samples data/raw/synthetic] [--engines paddle vlm]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_synthetic_samples as synth  # noqa: E402
from fin_extractor.config import load_config  # noqa: E402
from fin_extractor.extraction import markdown_to_lines  # noqa: E402
from fin_extractor.paths import DataPaths  # noqa: E402
from fin_extractor.preprocessing import preprocess  # noqa: E402


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("ํา", "ำ").replace("|", " ")
    return "".join(text.split())


def levenshtein(a: str, b: str) -> int:
    """ระยะแก้ไขแบบ DP สองแถว (ไม่พึ่งไลบรารีนอก pyproject)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference), normalize(hypothesis)
    return levenshtein(ref, hyp) / len(ref)


def reference_text() -> str:
    """ข้อความที่ render_page วาดลงหน้า (ไม่รวมตรายาง) เรียงตาม (y, x)."""
    drawn: list[tuple[float, float, str]] = []
    original = ImageDraw.ImageDraw.text

    def record(self, xy, text, *args, **kwargs):
        drawn.append((xy[1], xy[0], str(text)))
        return original(self, xy, text, *args, **kwargs)

    ImageDraw.ImageDraw.text = record
    try:
        synth.render_page(synth.INKS["black"], header_band=False, stamp=False)
    finally:
        ImageDraw.ImageDraw.text = original
    return "\n".join(s for _, _, s in sorted(drawn))


def read(path: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def make_reader(engine: str, cfg: dict):
    if engine == "paddle":
        from fin_extractor.ocr import create_engine

        reader = create_engine("paddle", cfg)
        return lambda img: reader.recognize(img).text
    from fin_extractor.vlm import TyphoonOCR

    reader = TyphoonOCR(cfg["vlm"])
    return lambda img: "\n".join(markdown_to_lines(reader.recognize(img).markdown))


def summarize(rows: list[dict]) -> dict:
    def stats(sub: list[dict]) -> dict:
        v = [r["cer"] for r in sub]
        return {"n": len(v), "mean": round(statistics.mean(v), 4), "median": round(statistics.median(v), 4),
                "max": round(max(v), 4)} if v else {"n": 0}

    return {
        "all": stats(rows),
        "no_stamp": stats([r for r in rows if not r["stamp"]]),
        "stamp": stats([r for r in rows if r["stamp"]]),
        "mean_seconds": round(statistics.mean(r["seconds"] for r in rows), 2) if rows else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="data/raw/synthetic")
    ap.add_argument("--engines", nargs="+", default=["paddle"], choices=["paddle", "vlm"])
    ap.add_argument("--out", help="ไฟล์ผล (ค่าตั้งต้น data/benchmarks/cer_synthetic_<engines>.json)")
    args = ap.parse_args()

    cfg = load_config()
    samples = Path(args.samples)
    truth = json.loads((samples / "truth.json").read_text(encoding="utf-8"))
    ref = reference_text()
    print(f"ข้อความอ้างอิง {len(normalize(ref))} ตัวอักษร (หลัง normalize)")

    result: dict = {"reference_chars": len(normalize(ref)), "engines": {}}
    for engine in args.engines:
        read_text = make_reader(engine, cfg)
        rows = []
        for name, meta in truth["files"].items():
            t0 = time.perf_counter()
            img = preprocess(read(samples / name), cfg["preprocessing"]).image
            hyp = read_text(img)
            row = {"file": name, "stamp": meta["stamp"], "scan_effects": meta["scan_effects"],
                   "cer": round(cer(ref, hyp), 4), "seconds": round(time.perf_counter() - t0, 2)}
            rows.append(row)
            print(f"[{engine}] {name:42} CER {row['cer']:.4f}  {row['seconds']:5.1f}s", flush=True)
        result["engines"][engine] = {"summary": summarize(rows), "files": rows}
        print(f"[{engine}] สรุป {json.dumps(result['engines'][engine]['summary'], ensure_ascii=False)}")

    out = Path(args.out or DataPaths().benchmarks() / f"cer_synthetic_{'-'.join(args.engines)}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved ->", out)


if __name__ == "__main__":
    main()
