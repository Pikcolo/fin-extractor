"""สร้างภาพย่อยพร้อมส่งเข้า OCR จากผล router: เฉพาะส่วนที่เก็บ ไฟล์ละ 2 แบบ (luma / flat)

    python -X utf8 scripts/make_ocr_sections.py data/raw/finance_doc_scan
    python -X utf8 scripts/make_ocr_sections.py data/raw/finance_doc_scan --only 1307

ลำดับงาน: notebook/lab (ผังแบ่ง) → make_clean_sections.py (ภาพย่อยทั้งหมด) → route_sections.py (อ่านคัดกรอง + ตัดสินใจ) → สคริปต์นี้
อินพุต: ภาพทั้งหน้าที่ preprocess แล้ว + ผังแบ่ง + ผล router (data/preprocess/<dataset>/<profile>/sections/router/<ไฟล์>.json)
ผล: data/preprocess/<dataset>/<profile>/sections/ocr/<ไฟล์>/
    sNN_luma.png / sNN_flat.png   เทา luma / V + พื้นหลังเรียบ (ให้ตัวอ่านอ่านทั้งสองแบบแล้วรวมตัวเลข ดู docs)
    index.json                    ส่วนที่เก็บตามลำดับอ่าน พร้อมบทบาทและเหตุผลที่ router เก็บ
กล่องครอบ (role=residual) ถูก **ลบส่วนย่อยทุกส่วนที่อยู่ข้างในออก (ทาขาว)** ก่อนทำสองแบบ: ส่วนย่อยถูกอ่านแยกอยู่แล้ว (หรือถูกตัดทิ้ง) จึงเหลือแต่ของนอกส่วนย่อย
เช่น เลขที่หนังสือรับรอง ป้ายยอดรวม
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
from fin_extractor.paths import dataset_of  # noqa: E402


def write_png(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError(f"encode ภาพไม่ได้: {path}")
    buf.tofile(str(path))


def mask_children(crop: np.ndarray, sec: dict, others: list[dict], work_shape: tuple[int, int], full_shape: tuple[int, int]) -> np.ndarray:
    """ทาขาวทับส่วนย่อยทุกส่วนที่อยู่ข้างใน ``sec`` (พิกัดเป็นหน่วย work ต้องคูณสัดส่วนกลับเป็นภาพเต็ม)."""
    fx, fy = full_shape[1] / work_shape[1], full_shape[0] / work_shape[0]
    cx, cy, cw, ch = sec["bbox"]
    out = crop.copy()
    for o in others:
        ox, oy, ow, oh = o["bbox"]
        ix0, iy0, ix1, iy1 = max(ox, cx), max(oy, cy), min(ox + ow, cx + cw), min(oy + oh, cy + ch)
        if ix1 > ix0 and iy1 > iy0 and (ix1 - ix0) * (iy1 - iy0) >= 0.9 * ow * oh:  # อยู่ข้างในจริง (≥ 90% ของพื้นที่ส่วนนั้น)
            out[int((iy0 - cy) * fy):int((iy1 - cy) * fy), int((ix0 - cx) * fx):int((ix1 - cx) * fx)] = 255
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--only", nargs="*", help="เลขท้ายชื่อไฟล์")
    args = ap.parse_args()
    dataset = dataset_of(args.input)
    stems = sorted(p.stem for p in L.PATHS.section_router(dataset).glob("*.json"))
    if args.only:
        stems = [s for s in stems if any(s.endswith(o) for o in args.only)]
    total_in = total_out = 0
    for stem in stems:
        doc = json.loads((L.PATHS.section_router(dataset) / f"{stem}.json").read_text(encoding="utf-8"))
        secs = doc["sections"]
        full = L.load_cached_preprocessed(stem, dataset=dataset)
        work, _ = L._resize_to_work(full)
        out = L.PATHS.section_ocr(dataset, stem)
        out.mkdir(parents=True, exist_ok=True)
        for old in out.glob("*"):
            old.unlink()
        index = []
        for s in secs:
            if not s["keep"]:
                continue
            crop = L.crop_full(full, work, s)
            if crop.size == 0:
                continue
            if s["role"] == "residual":
                inner = [o for o in secs if o is not s and o["role"] != "duplicate"]
                crop = mask_children(crop, s, inner, work.shape[:2], full.shape[:2])
            luma = L.enhance.to_gray(crop)
            flat = L.flatten_background(L.gray_value_channel(crop))
            write_png(out / f"{s['id']}_luma.png", luma)
            write_png(out / f"{s['id']}_flat.png", flat)
            index.append({"id": s["id"], "kind": s["kind"], "role": s["role"], "reason": s["reason"], "bbox": s["bbox"],
                          "size": [int(crop.shape[1]), int(crop.shape[0])], "files": {"luma": f"{s['id']}_luma.png", "flat": f"{s['id']}_flat.png"}})
        (out / "index.json").write_text(json.dumps({"source": stem, "page": 1, "sections": index}, ensure_ascii=False, indent=1), encoding="utf-8")
        total_in += len(secs)
        total_out += len(index)
        mb = sum(f.stat().st_size for f in out.glob("*.png")) / 1e6
        print(f"{stem[-4:]}: เก็บ {len(index)} จาก {len(secs)} ส่วน × 2 แบบ · {mb:.1f} MB", flush=True)
    print(f"รวม เก็บ {total_out} จาก {total_in} ส่วน ({total_out * 2} ภาพ) จาก {len(stems)} ไฟล์")


if __name__ == "__main__":
    main()
