"""จัดผล ``benchmark_real_scans.py`` ให้เป็นโฟลเดอร์ต่อ preprocessing หนึ่งชุด

อ่าน ``data/output/benchmarks/real_scans.json`` (ไม่รัน OCR ซ้ำ) แล้วเรนเดอร์ภาพของแต่ละชุดใหม่
เพื่อให้เปิดดูได้ว่าชุดนั้นทำอะไรกับภาพ

    data/output/benchmarks/preprocess_experiments/
      README.md            ตารางเทียบทุกชุด + คำอธิบายโฟลเดอร์
      <ชุด>/
        settings.json      ค่าที่ override + profile จริงที่ใช้
        metrics.csv        ตัวชี้วัดรายไฟล์
        images/            ภาพหลัง preprocess (JPEG q90 สำหรับดู; ขาวดำเป็น PNG)
        text/              ข้อความ OCR ของแต่ละไฟล์

    .venv/Scripts/python scripts/organize_preprocess_results.py [--only base,gray] [--no-images]
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_real_scans import VARIANTS  # noqa: E402
from fin_extractor.config import load_config  # noqa: E402
from fin_extractor.io import iter_input_files, load_pages, save_image  # noqa: E402
from fin_extractor.preprocessing import preprocess  # noqa: E402

DESCRIPTION = {
    "base": "ค่าตั้งต้นของโปรเจกต์: คงสี + orientation + deskew",
    "raw": "ไม่ preprocess เลย (ภาพที่ render จาก PDF ที่ 300 dpi)",
    "no_deskew": "ปิด deskew",
    "no_orient": "ปิดการหมุน 90/180/270°",
    "crop": "เปิด remove_borders (crop_to_content)",
    "gray": "แปลงเป็นภาพเทา",
    "median": "denoise: median",
    "nlmeans": "denoise: non-local means",
    "bilateral": "denoise: bilateral",
    "clahe": "เปิด CLAHE",
    "otsu": "ภาพเทา + binarize Otsu",
    "adaptive": "ภาพเทา + binarize adaptive",
    "sauvola": "ภาพเทา + binarize Sauvola",
}


def write_image(path: Path, img) -> None:
    if img.ndim == 2 and len(np.unique(img)) <= 2:  # ขาวดำล้วน: PNG เล็กและไม่เสียข้อมูล
        save_image(path.with_suffix(".png"), img)
    else:
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        path.with_suffix(".jpg").write_bytes(buf.tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/finance_doc_scan")
    ap.add_argument("--results", default="data/output/benchmarks/real_scans.json")
    ap.add_argument("--out", default="data/output/benchmarks/preprocess_experiments")
    ap.add_argument("--only", help="ชื่อชุดที่ต้องการ คั่นด้วย ,")
    ap.add_argument("--no-images", action="store_true", help="ข้ามการเรนเดอร์ภาพ")
    ap.add_argument(
        "--images-for",
        default="base,raw,no_deskew,no_orient",
        help="ชุดที่เก็บภาพ คั่นด้วย , (ชุดอื่นเก็บเฉพาะ settings/metrics/text เพราะผลวัดไม่ช่วย จึงไม่เก็บภาพ)",
    )
    args = ap.parse_args()
    image_sets = set(args.images_for.split(","))

    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    cfg = load_config()
    names = [n for n in results if not args.only or n in args.only.split(",")]
    out_root = Path(args.out)

    pages = {}
    if not args.no_images:
        for p in iter_input_files(args.input):
            pages[p.name] = load_pages(p, cfg["loader"]["pdf_dpi"])[0]

    for name in names:
        d = out_root / name
        (d / "text").mkdir(parents=True, exist_ok=True)
        override = VARIANTS.get(name)
        profile = None if override is None else {**copy.deepcopy(cfg["preprocessing"]), **override}
        (d / "settings.json").write_text(
            json.dumps(
                {"description": DESCRIPTION.get(name, ""), "override": override, "preprocessing": profile},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        rows = results[name]
        with open(d / "metrics.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["file", "lines", "conf", "valid_tax_ids", "amount_triple", "sec", "orientation", "skew_angle"])
            for fname, r in rows.items():
                m = r.get("meta", {})
                w.writerow([fname, r["lines"], f"{r['conf']:.3f}" if r["conf"] else "", " ".join(r["tax_ids"]),
                            r["triple"], r["sec"], m.get("orientation", ""), m.get("skew_angle", "")])
                (d / "text" / f"{Path(fname).stem}.txt").write_text(r["text"], encoding="utf-8")
        if pages and name in image_sets:
            (d / "images").mkdir(exist_ok=True)
            for fname, img in pages.items():
                proc = img if profile is None else preprocess(img, profile).image
                write_image(d / "images" / Path(fname).stem, proc)
        print(f"{name}: done", flush=True)

    n_docs = len(next(iter(results.values())))
    lines = [
        "# ผลทดลอง preprocessing บนสแกนจริง (`data/raw/finance_doc_scan`)",
        "",
        f"{n_docs} ไฟล์ · OCR = PaddleOCR (lang=th) · สร้างโดย `scripts/organize_preprocess_results.py` "
        "จากผลของ `scripts/benchmark_real_scans.py`",
        "",
        "ตัวชี้วัดทั้งหมด **ไม่ใช้เฉลย**: `tax_ids` = เลข 13 หลักที่ checksum ผ่าน, `triple` = พบชุด a+b=c โดย b≈VAT7% ของ a, "
        "`lines`/`conf` เป็นตัวชี้วัดอ่อน (ขึ้นได้แม้อ่านเป็นขยะ) ส่วนต่างระดับ 1–2 ใบจาก 18 ใบถือว่าอยู่ในความแกว่ง",
        "",
        "| ชุด | ทำอะไร | tax_ids (ใบที่เจอ) | triple | lines เฉลี่ย | conf เฉลี่ย | วินาที/หน้า |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in results:
        rs = list(results[name].values())
        n = len(rs)
        lines.append(
            f"| [`{name}`]({name}/) | {DESCRIPTION.get(name, '')} "
            f"| {sum(len(r['tax_ids']) for r in rs)} ({sum(bool(r['tax_ids']) for r in rs)}/{n}) "
            f"| {sum(r['triple'] for r in rs)}/{n} "
            f"| {sum(r['lines'] for r in rs) / n:.1f} "
            f"| {sum(r['conf'] or 0 for r in rs) / n:.3f} "
            f"| {sum(r['sec'] for r in rs) / n:.2f} |"
        )
    lines += [
        "",
        "## โครงสร้างโฟลเดอร์",
        "",
        "```",
        "<ชุด>/settings.json   ค่าที่ใช้     <ชุด>/metrics.csv   ตัวชี้วัดรายไฟล์",
        "<ชุด>/images/         ภาพหลัง preprocess (JPEG q90 เพื่อดู; OCR ไม่ได้อ่านจากไฟล์นี้) มีเฉพาะชุด "
        + ", ".join(f"`{s}`" for s in sorted(image_sets) if s in results)
        + " — ชุดอื่นไม่เก็บภาพ (สร้างใหม่ด้วย --images-for)",
        "<ชุด>/text/           ข้อความ OCR ดิบของแต่ละไฟล์",
        "```",
        "",
    ]
    (out_root / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"saved -> {out_root / 'README.md'}")


if __name__ == "__main__":
    main()
