"""สร้างผังแบ่ง section ของทั้งชุดในครั้งเดียว (ไม่ต้องเปิด notebook ทีละเล่ม) ขั้นภาพล้วน ไม่ใช้ OCR/GPU

    python -X utf8 -m fin_extractor preprocess data/raw/finance_doc_scan          # ต้องมีภาพที่ตรงแล้วก่อน
    python -X utf8 scripts/make_section_layouts.py data/raw/finance_doc_scan
    python -X utf8 scripts/make_section_layouts.py data/raw/finance_doc_scan --only 1309 1311 --table-mode cells
    python -X utf8 scripts/make_section_layouts.py data/raw/finance_doc_scan --max-piece-frac 0      # ไม่ตัดส่วนสูงเป็นกลุ่มแถว

อินพุต: ภาพทั้งหน้าที่ preprocess แล้ว (``preprocess/<dataset>/<profile>/final/``)
วิธี: ย่อเป็นสำเนาทำงานด้านยาว ``labkit.WORK`` → เทา → Otsu (blur 3) → ``labkit.layout_sections`` (ค่าเดียวกับที่ notebook ใช้)
ผล: ``preprocess/<dataset>/<profile>/sections/features/<ไฟล์>.json`` เขียนเฉพาะคีย์ ``layout``, ``layout_params``, ``layout_sections``
    (คีย์อื่นที่ notebook เคยบันทึกไว้คงเดิม) และภาพตรวจด้วยตา ``<ไฟล์>_layout.png``
    (เขียว = ช่องตารางที่เก็บ, น้ำเงิน = ตารางทั้งก้อน/ท่อน, ส้ม = ข้อความนอกตาราง, แดงหนา = ว่าง)
    ส่วนที่สูงเกิน ``--max-piece-frac`` ถูกตัดเป็นกลุ่มแถว มี ``part``/``parts`` ใน ``layout_sections`` (ท่อนหลังไม่มีหัวคอลัมน์ซ้ำ)
ตามด้วย make_clean_sections.py ได้เลย (อ่านผังจากที่เดียวกัน)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notebooks" / "preprocess_lab"))

import labkit as L  # noqa: E402
from fin_extractor.config import load_config  # noqa: E402
from fin_extractor.io import read_image  # noqa: E402
from fin_extractor.paths import DEFAULT_PROFILE, DataPaths, dataset_of  # noqa: E402
from fin_extractor.preprocessing import binarize as bin_ops  # noqa: E402


def write_png(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError(f"encode ภาพไม่ได้: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(path))


def page_images(final_dir: Path) -> dict[str, Path]:
    """ไฟล์ <ชื่อ>_p1.png ใต้ final/ (ทั้งแบบตรง ๆ และในโฟลเดอร์ย่อยต่อไฟล์) เฉพาะหน้า 1 เหมือนที่ขั้นอื่นของ lab ใช้."""
    return {p.name[: -len("_p1.png")]: p for p in sorted(final_dir.rglob("*_p1.png"))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="โฟลเดอร์ชุดข้อมูล (ใช้เอาชื่อ dataset)")
    ap.add_argument("--only", nargs="*", help="เลขท้ายชื่อไฟล์ เช่น 1309")
    ap.add_argument("--profile", default=DEFAULT_PROFILE, help="ชุดตั้งค่า preprocess ที่เป็นต้นทางของภาพ")
    ap.add_argument("--table-mode", choices=("whole", "cells", "bands", "rows"), default="rows",
                    help="rows (ค่าตั้งต้น) = แถบเต็มความกว้างตัดที่ขอบตารางและเส้นแนวนอนในตารางด้วย · whole = ตารางทั้งก้อนเป็นส่วนเดียว · cells = แตกเป็นช่องตามเส้น · bands = แถบเต็มความกว้างตัดที่ขอบตารางเท่านั้น")
    ap.add_argument("--max-piece-frac", type=float, default=L.MAX_PIECE_FRAC,
                    help="ส่วนที่สูงกว่านี้ (สัดส่วนของความสูงหน้า) ถูกตัดเป็นกลุ่มแถวที่แถวว่างระหว่างบรรทัด · 0 = ไม่ตัด")
    ap.add_argument("--config", help="ไฟล์ config ทับ default.yaml (เปลี่ยน paths.data_dir ได้)")
    args = ap.parse_args()

    paths = DataPaths(load_config(args.config))
    dataset = dataset_of(args.input)
    final_dir = paths.preprocess_final(dataset, args.profile)
    pages = page_images(final_dir) if final_dir.is_dir() else {}
    if not pages:
        sys.exit(f"ไม่พบภาพที่ preprocess แล้วใน {final_dir}\n"
                 f"รันก่อน: python -X utf8 -m fin_extractor preprocess {args.input} --profile {args.profile}")
    stems = [s for s in pages if not args.only or any(s.endswith(o) for o in args.only)]
    if not stems:
        sys.exit(f"ไม่มีไฟล์ที่ลงท้ายด้วย {args.only} ใน {final_dir}")

    out_dir = paths.section_features(dataset, args.profile)
    out_dir.mkdir(parents=True, exist_ok=True)
    totals = {"files": 0, "sections": 0, "kept": 0, "blank": 0}
    for stem in stems:
        full = read_image(pages[stem])
        work, _ = L._resize_to_work(full)
        bw = bin_ops.otsu(L.enhance.to_gray(work))
        secs = L.layout_sections(bw, table_mode=args.table_mode, max_piece_frac=args.max_piece_frac or None)
        kept = [s for s in secs if s["keep"]]
        by_kind = {k: sum(1 for s in secs if s["kind"] == k) for k in ("cell", "text", "table")}
        summary = {"tables": len({s["table"] for s in secs if s["table"] is not None}), "sections": len(secs),
                   "kept": len(kept), "dropped_blank": len(secs) - len(kept), "by_kind": by_kind}

        path = out_dir / f"{stem}.json"
        doc = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        doc.update(layout=summary, layout_sections=L.section_manifest(secs),
                   layout_params={"table_mode": args.table_mode, "max_piece_frac": args.max_piece_frac, "work_long_side": L.WORK,
                                  "page_size": [int(full.shape[1]), int(full.shape[0])],
                                  "work_size": [int(work.shape[1]), int(work.shape[0])]})
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

        overlay = L.draw_sections(work, secs)
        s = 1000 / max(overlay.shape[:2])
        write_png(out_dir / f"{stem}_layout.png", cv2.resize(overlay, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
        totals["files"] += 1
        totals["sections"] += len(secs)
        totals["kept"] += len(kept)
        totals["blank"] += len(secs) - len(kept)
        print(f"{stem[-4:]}: ตาราง {summary['tables']} · ส่วน {len(secs)} (เก็บ {len(kept)}, ว่าง {len(secs) - len(kept)}) "
              f"{by_kind}", flush=True)
    print(f"รวม {totals['files']} ไฟล์ · {totals['sections']} ส่วน (เก็บ {totals['kept']}, ว่าง {totals['blank']}) → {out_dir}")


if __name__ == "__main__":
    main()
