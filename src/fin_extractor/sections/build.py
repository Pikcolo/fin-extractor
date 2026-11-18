"""ขั้น 2–3 ของ flow: ผังแบ่ง section (``build_layouts``) และภาพรายส่วนก่อน OCR (``build_images``) — ภาพล้วน CPU ไม่เรียก OCR

ผลอยู่ที่ ``03_sections/<dataset>/<profile>/`` (ดู paths.py) ``layout/<ไฟล์>.json`` + ``<ไฟล์>_layout.png`` ภาพตรวจด้วยตา
(น้ำเงิน = ตารางทั้งก้อน/ท่อน, ส้ม = ข้อความ, เขียว = ช่อง (โหมด cells), แดง = ว่าง ไม่ส่ง OCR เลข = id ส่วน)
``images/<วิธี>/<ไฟล์>/sNN.png`` + ``index.json`` (วิธี = luma | flat | bin)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from ..io import read_image
from ..paths import DEFAULT_PROFILE, DataPaths
from ..preprocessing import binarize as bin_ops
from ..preprocessing import enhance
from . import images as I  # noqa: E741
from . import layout as L
from .crops import iter_section_crops


def write_png(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError(f"encode ภาพไม่ได้: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(path))


def page_images(final_dir: Path) -> dict[str, Path]:
    """ไฟล์ <ชื่อ>_p1.png ใต้ final/ (ทั้งแบบตรง ๆ และในโฟลเดอร์ย่อยต่อไฟล์) → {ชื่อไฟล์: พาธ}."""
    return {p.name[: -len("_p1.png")]: p for p in sorted(final_dir.rglob("*_p1.png"))}


def layout_params(table_mode: str, max_piece_frac: float | None, full: np.ndarray, work: np.ndarray) -> dict:
    return {"table_mode": table_mode, "max_piece_frac": max_piece_frac, "work_long_side": L.WORK,
            "page_size": [int(full.shape[1]), int(full.shape[0])], "work_size": [int(work.shape[1]), int(work.shape[0])]}


def build_layouts(paths: DataPaths, dataset: str, profile: str = DEFAULT_PROFILE, only: list[str] | None = None,
                  table_mode: str = "rows", max_piece_frac: float | None = L.MAX_PIECE_FRAC) -> dict[str, int]:
    """ผังแบ่ง section ของทุกไฟล์ใน ``02_preprocess/<dataset>/<profile>/final/`` (กรองด้วยท้ายชื่อใน ``only``)
    เขียนเฉพาะคีย์ ``layout`` ``layout_params`` ``layout_sections`` คีย์อื่นที่ notebook เคยบันทึกไว้คงเดิม คืนยอดรวม."""
    final_dir = paths.preprocess_final(dataset, profile)
    pages = page_images(final_dir) if final_dir.is_dir() else {}
    if not pages:
        raise FileNotFoundError(f"ไม่พบภาพที่ preprocess แล้วใน {final_dir} — รัน: python -X utf8 -m fin_extractor preprocess <input> --profile {profile}")
    stems = [s for s in pages if not only or any(s.endswith(o) for o in only)]
    if not stems:
        raise FileNotFoundError(f"ไม่มีไฟล์ที่ลงท้ายด้วย {only} ใน {final_dir}")

    out_dir = paths.section_layout(dataset, profile)
    out_dir.mkdir(parents=True, exist_ok=True)
    totals = {"files": 0, "sections": 0, "kept": 0, "blank": 0}
    for stem in stems:
        full = read_image(pages[stem])
        work, _ = L.resize_to_work(full)
        bw = bin_ops.otsu(enhance.to_gray(work))
        secs = L.layout_sections(bw, table_mode=table_mode, max_piece_frac=max_piece_frac or None)
        kept = [s for s in secs if s["keep"]]
        by_kind = {k: sum(1 for s in secs if s["kind"] == k) for k in ("cell", "text", "table")}
        summary = {"tables": len({s["table"] for s in secs if s["table"] is not None}), "sections": len(secs),
                   "kept": len(kept), "dropped_blank": len(secs) - len(kept), "by_kind": by_kind}

        path = out_dir / f"{stem}.json"
        doc = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        doc.update(layout=summary, layout_sections=[dict(s) for s in secs], layout_params=layout_params(table_mode, max_piece_frac, full, work))
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

        overlay = L.draw_sections(work, secs)
        s = 1000 / max(overlay.shape[:2])
        write_png(out_dir / f"{stem}_layout.png", cv2.resize(overlay, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
        totals["files"] += 1
        totals["sections"] += len(secs)
        totals["kept"] += len(kept)
        totals["blank"] += len(secs) - len(kept)
        print(f"{stem[-4:]}: ตาราง {summary['tables']} · ส่วน {len(secs)} (เก็บ {len(kept)}, ว่าง {len(secs) - len(kept)}) {by_kind}", flush=True)
    print(f"รวม {totals['files']} ไฟล์ · {totals['sections']} ส่วน (เก็บ {totals['kept']}, ว่าง {totals['blank']}) → {out_dir}")
    return totals


def build_images(paths: DataPaths, dataset: str, only: list[str] | None = None, drop_empty_rows: bool = False,
                 variants: tuple[str, ...] = I.VARIANTS) -> int:
    """ภาพรายส่วนของทุกไฟล์ที่มีผัง แยกโฟลเดอร์ตามวิธี (``variants``) เขียน ``index.json`` ต่อวิธี คืนจำนวน section ทั้งหมด.
    ``drop_empty_rows``: แบบ bin ทาดำแถวตารางที่ช่องตัวเลขว่าง (ฟอร์มพิมพ์รายการล่วงหน้า เช่น 50 ทวิ) ส่วนที่ทาดำหมดไม่มีไฟล์ (``file: null``)."""
    stems = sorted(p.stem for p in paths.section_layout(dataset).glob("*.json"))
    if only:
        stems = [s for s in stems if any(s.endswith(o) for o in only)]
    total = 0
    images_root = paths.section_images(dataset)
    for stem in stems:
        crops = [(sec, crop) for sec, crop in iter_section_crops(paths, stem, dataset) if crop.size]
        bins = [(sec, I.binarize_inverted(crop)) for sec, crop in crops]
        if drop_empty_rows:
            dropped = I.drop_empty_rows_by_table(bins)
            infos = [info for _img, info in dropped]
            bins = [(sec, img) for (sec, _), (img, _info) in zip(bins, dropped)]
        else:
            infos = [{} for _ in bins]
        made = {"luma": lambda: [enhance.to_gray(c) for _, c in crops],
                "flat": lambda: [I.flatten_background(I.gray_value_channel(c)) for _, c in crops],
                "bin": lambda: [img for _, img in bins]}
        mb = 0.0
        for variant in variants:
            out = paths.section_images(dataset, stem, variant)
            if out.exists():
                shutil.rmtree(out)
            out.mkdir(parents=True)
            index = []
            for (sec, crop), img, info in zip(crops, made[variant](), infos):
                blank = variant == "bin" and info.get("blank")  # ทาดำหมดทั้งส่วน = ไม่มีไฟล์ ตัวอ่านข้าม (ยังอยู่ใน index ให้ตรงกับผัง)
                if not blank:
                    write_png(out / f"{sec['id']}.png", img)
                index.append({"id": sec["id"], "kind": sec["kind"], "bbox": sec["bbox"], "table": sec["table"],
                              **({"part": sec["part"], "parts": sec["parts"]} if sec.get("part") else {}),
                              "size": [int(crop.shape[1]), int(crop.shape[0])], "file": None if blank else f"{sec['id']}.png",
                              **({"rows": info} if variant == "bin" and info.get("applied") else {})})
            (out / "index.json").write_text(json.dumps({"source": stem, "page": 1, "variant": variant, "sections": index}, ensure_ascii=False, indent=1), encoding="utf-8")
            mb += sum(f.stat().st_size for f in out.glob("*.png")) / 1e6
        total += len(crops)
        print(f"{stem[-4:]}: {len(crops)} section × {len(variants)} วิธี · {mb:.1f} MB → {images_root}/<วิธี>/{stem}", flush=True)
    print(f"รวม {total} section จาก {len(stems)} ไฟล์")
    return total
