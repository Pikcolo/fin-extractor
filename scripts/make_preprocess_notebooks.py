"""สร้าง notebook รายไฟล์สำหรับเช็คการ preprocess (edge detection / threshold / การแบ่ง section) ไฟล์ละ 1 เล่ม

    python scripts/make_preprocess_notebooks.py data/01_input/finance_doc_scan
    python scripts/make_preprocess_notebooks.py data/01_input/finance_doc_scan --out notebooks/preprocess_lab/finance_doc_scan

เครื่องมือวิเคราะห์อยู่ที่ notebooks/preprocess_lab/labkit.py (notebook เรียกใช้ ไม่มีโค้ดซ้ำ) ถ้ามีไฟล์เพิ่ม รันสคริปต์ซ้ำได้
notebook ที่สร้างไม่มี output (สะอาด) เพราะภาพสแกนเป็นข้อมูลภาษีจริง — .gitignore ไม่ให้ commit *.ipynb ในโฟลเดอร์นี้
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fin_extractor.io import iter_input_files  # noqa: E402
from fin_extractor.paths import dataset_of  # noqa: E402

INTRO = """# Preprocess lab: __STEM__

ไฟล์: `__REL__`

เป้าหมาย: ดูว่าใบนี้ควรปรับภาพแบบไหนก่อนส่งเข้าตัวอ่าน เพื่อให้ router เลือกวิธีตามลักษณะภาพได้
เทียบ 3 เรื่อง คือ **edge detection**, **threshold** และ **การแบ่งภาพเป็น section (contour)**

หมายเหตุ
- วิเคราะห์บนภาพที่ **ปรับให้ตรงแล้ว** (แก้ทิศ + deskew ใช้ภาพที่ pipeline เก็บไว้ถ้ามี) และย่อเป็นสำเนาขนาดคงที่ (ด้านยาว 1600 พิกเซล) เพื่อให้ตัวเลขเทียบข้ามไฟล์ได้
- ค่าที่กวาด (kernel, gap, threshold) เป็น **ค่าทดลอง** ไม่ใช่ค่าตั้งต้นของระบบ และยังไม่มีเฉลยว่าแบบไหน "ถูก" ตัวเลขที่ได้บอกแค่ปริมาณ (หมึกกี่ %, กี่ชิ้น, กี่กรอบ) ต้องดูภาพประกอบ
- ท้าย notebook มีช่องบันทึกผลที่คุณตัดสินด้วยตา (`VERDICT`) เก็บลง JSON ไว้ให้ router เรียนรู้จากหลายไฟล์
"""

LOAD = """import sys
from pathlib import Path

for p in (Path.cwd().parent, Path.cwd() / "notebooks" / "preprocess_lab"):  # labkit.py อยู่ notebooks/preprocess_lab/ (รันจากโฟลเดอร์ชุดข้อมูล หรือจาก root โปรเจกต์)
    sys.path.insert(0, str(p))
import labkit as L

STEM = "__STEM__"
SRC = L.ROOT / "__REL__"
full, work, scale = L.load(SRC)

print(f"ภาพเต็ม {full.shape[1]}×{full.shape[0]} → สำเนาทำงาน {work.shape[1]}×{work.shape[0]} (ย่อ {scale:.3f})")
print(f"ที่มาของภาพ: {L.STATE['source']} · มุมเอียงก่อน {L.STATE['skew_before_deg']:+.1f}° → หลัง {L.STATE['skew_after_deg']:+.1f}°")
L.grid({"ภาพที่ใช้วิเคราะห์ (ปรับตรงแล้ว)": work}, cols=1)"""

FEATURES = """feats = L.features(work)
import pandas as pd
pd.Series(feats, name=STEM).to_frame()"""

EDGES = """gray = L.enhance.to_gray(work)
edge_imgs, edge_stats = L.edges(gray)
L.grid({k: 255 - v for k, v in edge_imgs.items()}, cols=3,
       notes={k: f"edge density {s['edge_density']:.3f}" for k, s in edge_stats.items()},
       title="edge detection (เส้นดำ = edge)")
L.table(edge_stats)"""

THRESH = """thr_imgs, thr_stats = L.thresholds(gray)
L.grid(thr_imgs, cols=3,
       notes={k: f"หมึก {s['ink_ratio']:.3f} · {s['components']} ชิ้น · ชิ้นเล็ก {s['speck_ratio_le4px']:.2f}" for k, s in thr_stats.items()},
       title="threshold (เทียบวิธี)")
L.table(thr_stats)"""

SECTIONS_MD = """## 4. แบ่งภาพเป็น section

เลือกภาพขาวดำที่ใช้เป็นฐานด้านล่าง (`BASE_THRESHOLD`) แล้วดู 3 วิธีแบ่ง:
1. **ขยายหมึกแล้วหา contour**: kernel เล็ก = ระดับตัวอักษร/คำ, ใหญ่ = บล็อก/ส่วนใหญ่ ดูว่ากรอบที่ได้ตรงกับ "ส่วน" ของเอกสารที่คุณอยากส่งอ่านทีละส่วนที่ขนาดไหน
2. **ตัดตามแถวว่างแนวนอน** (แถบ): ง่ายกว่า contour และไม่ตัดกลางบรรทัด
3. **ช่องของตารางจากเส้น**: นับเส้นแนวนอน/แนวตั้งและช่องที่ล้อมด้วยเส้น"""

SECTIONS = """BASE_THRESHOLD = "otsu (blur 3)"   # เปลี่ยนเป็นชื่ออื่นใน thr_imgs ได้ ถ้าวิธีนั้นดูดีกว่าในหัวข้อ 3
bw = thr_imgs[BASE_THRESHOLD]

box_sets = {f"{kx}×{ky}": L.boxes_by_dilation(bw, (kx, ky)) for kx, ky in L.KERNELS}
L.grid({f"dilate {k}": L.draw_boxes(work, b) for k, b in box_sets.items()}, cols=3,
       notes={f"dilate {k}": f"{len(b)} กรอบ" for k, b in box_sets.items()}, title="contour หลังขยายหมึก (kernel กว้าง×สูง)")

band_sets = {g: L.bands_by_gaps(bw, g) for g in L.BAND_GAPS}
L.grid({f"แถวว่าง ≥ {g}px": L.draw_bands(work, b) for g, b in band_sets.items()}, cols=3,
       notes={f"แถวว่าง ≥ {g}px": f"{len(b)} แถบ" for g, b in band_sets.items()}, title="แบ่งเป็นแถบแนวนอน")

cells = L.cells_from_lines(bw)
horiz, vert = L.line_masks(bw)
L.grid({"เส้นแนวนอน": 255 - horiz, "เส้นแนวตั้ง": 255 - vert, "ช่องตารางที่ล้อมด้วยเส้น": L.draw_boxes(work, cells, (0, 160, 0), 2)},
       cols=3, notes={"ช่องตารางที่ล้อมด้วยเส้น": f"{len(cells)} ช่อง"}, title="ตารางจากเส้น")

sections_summary = {
    "base_threshold": BASE_THRESHOLD,
    "boxes_by_kernel": {k: len(b) for k, b in box_sets.items()},
    "bands_by_min_gap": {str(g): len(b) for g, b in band_sets.items()},
    "table_cells": len(cells),
}
sections_summary"""

TABLE_MD = """## 5. แบ่งตามตาราง → ภาพย่อย → เช็ค preprocess ราย section

แนวคิด: ถ้าหน้ามีตาราง ให้แตกเป็นภาพย่อยตามช่อง (cell) ส่วนที่อยู่นอกตารางเป็นแถบข้อความ แล้วดูทีละส่วนว่า
- ส่วนไหน **ว่าง** (ไม่มีชิ้นหมึกเลย) → ตัดทิ้ง ไม่ต้องส่งเข้า OCR
- ส่วนที่เก็บ ควร **ทำ grayscale / ลด noise** แบบไหนก่อนเข้าโมเดล

ผังนี้เป็นขั้น preprocess ล้วน ยังไม่มีการอ่านข้อความ ภาพย่อยไม่เก็บเป็นไฟล์ ตัดจากภาพทั้งหน้าตามผังตอนใช้ (`L.iter_section_crops`) ได้ภาพเดียวกันทุกครั้ง
ค่าตัดสิน (ขนาดตารางขั้นต่ำ, ขนาดชิ้นหมึกที่ไม่นับเป็นจุดรบกวน) เป็นค่าทดลอง ดูตารางสรุปประกอบภาพก่อนเชื่อ"""

TABLE_LAYOUT = """secs = L.layout_sections(bw)
kept = [s for s in secs if s["keep"]]
n_tables = len({s["table"] for s in secs if s["table"] is not None})
print(f"พบตาราง {n_tables} ตาราง · แบ่งได้ {len(secs)} ส่วน · เก็บ {len(kept)} · ตัดทิ้ง (ว่าง) {len(secs) - len(kept)}")
L.grid({"ผังการแบ่ง (เขียว = ช่องตารางที่เก็บ, ส้ม = ข้อความนอกตาราง, แดงหนา = ว่าง ตัดทิ้ง)": L.draw_sections(work, secs)}, cols=1)
L.table({s["id"]: {"kind": s["kind"], "table": s["table"], "x": s["bbox"][0], "y": s["bbox"][1], "w": s["bbox"][2],
                   "h": s["bbox"][3], "ink_ratio": s["ink_ratio"], "components": s["components"], "keep": s["keep"]} for s in secs})"""

TABLE_VARIANTS = """# ดู preprocess ก่อนเข้า OCR กับ 3 ส่วนที่มีหมึกมากที่สุด (ส่วนที่ keep) ตัวเลขใต้ภาพ: noise / contrast / ink
top = sorted(kept, key=lambda s: -s["ink_ratio"] * s["bbox"][2] * s["bbox"][3])[:3]
variant_stats = {}
for s in top:
    imgs, stats = L.variants(L.crop_full(full, work, s))
    variant_stats[s["id"]] = stats
    L.grid(imgs, cols=3, title=f"{s['id']} ({s['kind']}, {s['bbox'][2]}×{s['bbox'][3]} บนสำเนาทำงาน)",
           notes={k: f"noise {v['noise']:.2f} · contrast {v['contrast']:.0f} · ink {v['ink']:.3f}" for k, v in stats.items()})"""

TABLE_SAVE = """manifest = L.section_manifest(secs)
layout_summary = {"tables": n_tables, "sections": len(secs), "kept": len(kept), "dropped_blank": len(secs) - len(kept),
                  "by_kind": {k: sum(1 for s in secs if s["kind"] == k) for k in ("cell", "text", "table")}}
print(f"บันทึกผังแบ่ง {len(manifest)} ส่วน (เก็บ {sum(1 for m in manifest if m['keep'])}) ภาพย่อยตัดจากภาพทั้งหน้าตอนใช้ด้วย L.iter_section_crops ไม่เก็บเป็นไฟล์")
layout_summary"""

SAVE = """L.save(STEM, features=feats, edges=edge_stats, thresholds=thr_stats, sections=sections_summary,
       layout=layout_summary, layout_sections=manifest, section_variants=variant_stats)
L.compare_with_others(STEM, keys=["ink_ratio", "contrast_p95_p5", "noise_median_diff", "skew_deg",
                                  "color_saturation", "hlines", "vlines", "components"])"""

VERDICT_MD = """## 7. ตัดสินด้วยตา (เติมเอง)

ดูภาพด้านบนแล้วบันทึกว่าใบนี้วิธีไหน "เวิร์ค" เก็บลง JSON ไว้ ถ้าหลายไฟล์ที่ features คล้ายกันเลือกวิธีเดียวกัน
นั่นคือสัญญาณให้ router ใช้กฎนั้นกับภาพลักษณะแบบนี้ (ตัวเลขเกณฑ์ต้องมาจากข้อมูลรวม ไม่ใช่ตั้งเอง)"""

VERDICT = """VERDICT = {
    "threshold": None,   # เช่น "sauvola 25/0.2" หรือ "ไม่ต้อง threshold (ภาพสีอ่านดีกว่า)"
    "edge": None,        # เช่น "canny otsu-based" หรือ "ไม่ใช้"
    "sections": None,    # เช่น "dilate 140×20" หรือ "ไม่แบ่ง ส่งทั้งแผ่น"
    "table_split": None,   # แบ่งตามตารางในหัวข้อ 5 ได้ดีไหม เช่น "ดี" / "ตัดช่องผิด" / "ไม่มีตาราง"
    "per_section_preprocess": None,  # เช่น "เทา + median" หรือ "สีเดิม"
    "drop": None,        # ส่วนที่ควรตัดทิ้งเพิ่ม เช่น "s12 บล็อกเงื่อนไขท้ายใบ"
    "note": "",          # สิ่งที่เห็น: เงา, เอียง, ตัวอักษรอ่อน, ตาราง, ลายมือ ฯลฯ
}
L.save(STEM, verdict=VERDICT)"""


def build(path: Path, rel: str) -> nbformat.NotebookNode:
    def fill(src: str) -> str:
        return src.replace("__STEM__", path.stem).replace("__REL__", rel)

    cells = [
        new_markdown_cell(fill(INTRO)),
        new_code_cell(fill(LOAD)),
        new_markdown_cell("## 1. คุณลักษณะของภาพ (ฟีเจอร์ที่ router ใช้ตัดสิน)\n\nขนาด, พื้นหลัง/คอนทราสต์, สัดส่วนหมึก, สี, สัญญาณรบกวน, มุมเอียง, ชิ้นส่วนเล็ก ๆ, เส้นตาราง"),
        new_code_cell(FEATURES),
        new_markdown_cell("## 2. Edge detection\n\nCanny (สามแบบตั้ง threshold), Sobel, Laplacian, morphological gradient ดูว่าวิธีไหนจับขอบตัวอักษร/เส้นตารางได้สะอาด และวิธีไหนจับสัญญาณรบกวนมาด้วย"),
        new_code_cell(EDGES),
        new_markdown_cell("## 3. Threshold\n\nเทียบ Otsu, Otsu+CLAHE, adaptive และ Sauvola ดูตัวอักษรจาง/เงา/พื้นหลังเปื้อน ตัวเลข \"ชิ้นเล็ก\" คือสัดส่วนชิ้นส่วนที่เล็กมาก (≤4 พิกเซล) เป็นตัวบอกว่าติดสัญญาณรบกวนมากแค่ไหน"),
        new_code_cell(THRESH),
        new_markdown_cell(SECTIONS_MD),
        new_code_cell(SECTIONS),
        new_markdown_cell(TABLE_MD),
        new_code_cell(TABLE_LAYOUT),
        new_code_cell(TABLE_VARIANTS),
        new_code_cell(TABLE_SAVE),
        new_markdown_cell("## 6. บันทึกผล และเทียบกับไฟล์อื่น\n\nเขียนผลลง `data/03_sections/<dataset>/<profile>/layout/<ไฟล์>.json` แล้วเทียบ features ของใบนี้กับใบอื่นที่รันแล้ว (เปอร์เซ็นไทล์)"),
        new_code_cell(SAVE),
        new_markdown_cell(VERDICT_MD),
        new_code_cell(VERDICT),
    ]
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return nb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="ไฟล์หรือโฟลเดอร์ของภาพ/PDF")
    ap.add_argument("--out", help="โฟลเดอร์ปลายทาง (ค่าตั้งต้น notebooks/preprocess_lab/<ชื่อชุดข้อมูล>/)")
    args = ap.parse_args()

    out = Path(args.out) if args.out else ROOT / "notebooks" / "preprocess_lab" / dataset_of(args.input)
    out.mkdir(parents=True, exist_ok=True)
    files = list(iter_input_files(args.input))
    for f in files:
        f = f.resolve()
        rel = f.relative_to(ROOT).as_posix() if f.is_relative_to(ROOT) else f.as_posix()
        nbformat.write(build(f, rel), out / f"{f.stem}.ipynb")
    print(f"สร้าง {len(files)} notebook ที่ {out}")


if __name__ == "__main__":
    main()
