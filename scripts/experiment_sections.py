"""ทดลองแบ่งหน้าเป็นท่อนก่อนส่งเข้า OpenThai: วิธีไหนกู้ค่าที่ส่งทั้งแผ่นทิ้งไปได้

    python -X utf8 scripts/experiment_sections.py                # ชุดเริ่มต้น (ใบที่เคย failed + ใบเทียบ)
    python -X utf8 scripts/experiment_sections.py 1307 1321 --methods strips3 snap3

วิธี (ทำบนภาพที่ pipeline preprocess แล้ว = ตรงแล้ว)
- whole   : ส่งทั้งแผ่น (ใช้ข้อความดิบที่เก็บไว้ ไม่รันใหม่)
- strips2 / strips3 : แบ่งเป็นแถบแนวนอนเท่า ๆ กัน 2 / 3 แถบ ซ้อนทับกันเล็กน้อยกันตัดกลางบรรทัด
- snap2 / snap3     : ตัดที่ "ช่องว่างระหว่างบรรทัด" ที่ใกล้ตำแหน่งแบ่งเท่า ๆ กันที่สุด (ไม่ซ้อนทับ ไม่ตัดกลางข้อความ)

วัดผลด้วย rules ล้วน (ไม่ใช้ LLM เพื่อให้เห็นผลของการอ่านล้วน ๆ): สถานะ, ฟิลด์ที่ขาด, ค่าที่ไม่พบในข้อความ, ค่า key ของใบนั้นมีไหม
ไม่มี ground truth จึงวัดความ "ครบ" ไม่ได้วัดความ "ถูก" ผลลง data/etc/experiments/openthai_sections/<วิธี>/ (ไม่ถูก commit)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notebooks" / "preprocess_lab"))

import labkit as L  # noqa: E402
from fin_extractor.config import load_config  # noqa: E402
from fin_extractor.io import iter_input_files  # noqa: E402
from fin_extractor.paths import DataPaths, dataset_of  # noqa: E402
from fin_extractor.pipeline import FinanceOCRPipeline  # noqa: E402
from fin_extractor.stages import load_raw, prepared_pages, raw_pages  # noqa: E402

DEFAULT_STEMS = ["1307", "1321", "1323", "1309", "1311", "1310", "1313"]  # failed ×4 + 50 ทวิ ที่ผ่าน + ใบกำกับภาษี 2 ใบเทียบ
OVERLAP = 0.05  # สัดส่วนความสูงหน้าที่แถบซ้อนกัน (ทดลอง: ให้ทุกบรรทัดอยู่ครบในอย่างน้อยหนึ่งแถบ)
SNAP_WINDOW = 0.10  # ตัดที่ช่องว่างได้ห่างจากตำแหน่งเท่า ๆ กันไม่เกินสัดส่วนนี้ของความสูงหน้า (ทดลอง)
MIN_GAP = 16  # ความสูงแถวว่างขั้นต่ำที่ WORK=1600 (ค่าเดียวกับ lab)


def strips(height: int, n: int) -> list[tuple[int, int]]:
    ov = int(OVERLAP * height)
    cuts = [round(k * height / n) for k in range(n + 1)]
    return [(max(0, cuts[k] - (ov if k else 0)), min(height, cuts[k + 1] + (ov if k < n - 1 else 0))) for k in range(n)]


def snapped(img: np.ndarray, n: int) -> list[tuple[int, int]]:
    """ตัดที่ช่องว่างระหว่างบรรทัดใกล้ตำแหน่งแบ่งเท่า ๆ กัน (ไม่มีช่องว่างในช่วงที่ยอมให้ → ตัดตรงตำแหน่งเท่า ๆ กัน)."""
    h = img.shape[0]
    work, scale = L._resize_to_work(img)
    gray = L.enhance.to_gray(work)
    bw = L.bin_ops.otsu(gray)
    bands = L.bands_by_gaps(bw, MIN_GAP)
    gaps = [((b0[1] + b1[0]) / 2) / scale for b0, b1 in zip(bands, bands[1:])]  # กึ่งกลางช่องว่างระหว่างแถบ (พิกเซลภาพเต็ม)
    cuts = [0]
    for k in range(1, n):
        ideal = k * h / n
        near = [g for g in gaps if abs(g - ideal) <= SNAP_WINDOW * h]
        cuts.append(int(min(near, key=lambda g: abs(g - ideal))) if near else int(ideal))
    cuts.append(h)
    return list(zip(cuts[:-1], cuts[1:]))


def pieces(method: str, img: np.ndarray) -> list[tuple[int, int]]:
    n = int(method[-1])
    return strips(img.shape[0], n) if method.startswith("strips") else snapped(img, n)


def evaluate(pipe: FinanceOCRPipeline, source: str, text: str) -> dict:
    r = pipe.process_raw(source, 1, text)
    profile = pipe.profile_for(r.data)
    key = profile.primary_amount or "total_amount"
    flat_has = r.field_evidence.get(key)
    return {
        "status": r.status, "missing": len(r.missing_fields),
        "ungrounded": sum(1 for e in r.field_evidence.values() if e.grounded is False),
        "key_field": key.replace("document_fields.", ""), "key_present": bool(flat_has and flat_has.value),
        "key_value": flat_has.value if flat_has else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stems", nargs="*", default=DEFAULT_STEMS, help="เลขท้ายชื่อไฟล์ เช่น 1307")
    ap.add_argument("--methods", nargs="+", default=["strips2", "strips3", "snap2", "snap3"])
    ap.add_argument("--input", default=str(ROOT / "data" / "raw" / "finance_doc_scan"))
    ap.add_argument("--out", help="ค่าตั้งต้น = paths.experiments(\"openthai_sections\")")
    args = ap.parse_args()

    cfg = load_config()
    paths = DataPaths(cfg)
    files = {f.stem[-4:]: f for f in iter_input_files(args.input)}
    dataset = dataset_of(args.input)
    pdir = paths.preprocess(dataset, "default")
    raw_dir = paths.raw_data("openthai", dataset)
    pipe = FinanceOCRPipeline(cfg, mode="vlm", engine="openthai", use_llm=False, with_reader=False)

    from fin_extractor.vlm.openthai_ocr import OpenThaiOCR

    reader = OpenThaiOCR(cfg["openthai"])
    out_root = Path(args.out) if args.out else paths.experiments("openthai_sections")
    out_root.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for s in args.stems:
        f = files[s]
        img = next(img for _, img, _ in prepared_pages(f, cfg, pdir))
        base = raw_pages(raw_dir, f.stem, "md")
        base_text = load_raw(base[0][1])[0] if base else ""
        results.append({"file": s, "method": "whole", "pieces": 1, "seconds": None, **evaluate(pipe, str(f), base_text)})
        for m in args.methods:
            cuts = pieces(m, img)
            t0 = time.perf_counter()
            parts = [reader.recognize(img[y0:y1]).markdown for y0, y1 in cuts]
            sec = round(time.perf_counter() - t0, 1)
            text = "\n\n".join(parts)
            d = out_root / m
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{f.stem}_p1.md").write_text(text, encoding="utf-8")
            results.append({"file": s, "method": m, "pieces": len(cuts), "seconds": sec, **evaluate(pipe, str(f), text)})
            r = results[-1]
            print(f"{s} {m:8s} {len(cuts)} ท่อน {sec:5.1f}s  {r['status']:12s} ขาด={r['missing']} ไม่พบในข้อความ={r['ungrounded']} "
                  f"{r['key_field']}={r['key_value']}", flush=True)

    import pandas as pd

    df = pd.DataFrame(results)
    df.to_csv(out_root / "summary.csv", index=False, encoding="utf-8")
    print("\n" + df.to_string(index=False))
    print("\nรวมต่อวิธี (ทุกไฟล์):")
    print(df.groupby("method").agg(files=("file", "count"), failed=("status", lambda x: (x == "failed").sum()),
                                   missing=("missing", "sum"), ungrounded=("ungrounded", "sum"),
                                   key_present=("key_present", "sum"), seconds=("seconds", "sum")).to_string())


if __name__ == "__main__":
    main()
