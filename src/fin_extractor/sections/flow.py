"""flow หลักห้าขั้น (``python -m fin_extractor sections``): ตัดสินว่าขั้นไหนของแต่ละไฟล์ "ยังใช้ได้" จึงข้ามได้

    preprocess → layout → images → ocr → extract
    ภาพ final     ผัง      ภาพรายส่วน  ข้อความ+ที่มา   JSON ตาม schema

กติกาข้ามของแต่ละขั้น (เทียบกับผลของขั้นก่อนหน้าเท่านั้น ไม่เดา)
- layout: มี ``layout/<ไฟล์>.json`` ที่พารามิเตอร์ (table_mode, max_piece_frac, fill_text_ink, dotted_rules, mask_codes, trim_sections) ตรงกับที่สั่ง และใหม่กว่าภาพ final
- images: ``images/<วิธี>/<ไฟล์>/index.json`` ตรงกับผังปัจจุบัน (id + bbox ของส่วนที่เก็บ) และใหม่กว่าไฟล์ผัง
- ocr:    ``.sections.json`` มีลายนิ้วมือผังตรงและ ``complete`` (ตรวจใน ``read_dataset``)
- extract: ทำใหม่ทุกครั้ง (CPU เร็ว ไม่โหลดโมเดล)
"""

from __future__ import annotations

import json
from pathlib import Path

from ..paths import DEFAULT_PROFILE, DataPaths
from .build import page_images
from .read import kept_layout

STAGES = ("preprocess", "layout", "images", "ocr", "extract")


def layout_is_current(paths: DataPaths, dataset: str, stem: str, table_mode: str, max_piece_frac: float | None,
                      profile: str = DEFAULT_PROFILE, fill_text: bool = False, dotted: bool = False, mask_codes: bool = True,
                      trim: str | None = None) -> bool:
    path = paths.section_layout(dataset, profile) / f"{stem}.json"
    page = page_images(paths.preprocess_final(dataset, profile)).get(stem)
    if not path.exists() or page is None or path.stat().st_mtime < page.stat().st_mtime:
        return False
    params = json.loads(path.read_text(encoding="utf-8")).get("layout_params", {})
    return (params.get("table_mode") == table_mode and params.get("max_piece_frac") == max_piece_frac
            and bool(params.get("fill_text_ink", False)) == fill_text and bool(params.get("dotted_rules", False)) == dotted  # ผังเก่าไม่มีคีย์ = แบบเดิม (ไม่คำนึงพื้นสี)
            and bool(params.get("mask_codes", False)) == mask_codes  # ผังก่อนมีการระบาย QR ไม่มีคีย์ = ไม่ได้ระบาย
            and (params.get("trim_sections") or False) == (trim or False))  # ไม่หด = False ในผัง


def images_are_current(paths: DataPaths, dataset: str, stem: str, variants: tuple[str, ...], profile: str = DEFAULT_PROFILE) -> bool:
    layout_file = paths.section_layout(dataset, profile) / f"{stem}.json"
    if not layout_file.exists():
        return False
    layout = json.loads(layout_file.read_text(encoding="utf-8"))["layout_sections"]
    for variant in variants:
        index = paths.section_images(dataset, stem, variant, profile) / "index.json"
        if not index.exists() or index.stat().st_mtime < layout_file.stat().st_mtime:
            return False
        got = [(s["id"], list(s["bbox"])) for s in json.loads(index.read_text(encoding="utf-8"))["sections"]]
        if got != kept_layout(layout):
            return False
    return True


def orientation_flags(paths: DataPaths, dataset: str, stems: list[str], profile: str = DEFAULT_PROFILE) -> list[tuple[str, str]]:
    """[(ไฟล์, ข้อผิดพลาด)] ของหน้าที่ตรวจทิศไม่ได้ตอน preprocess (``orientation_error`` ใน ``<ไฟล์>_p1.meta.json``) ภาพอาจตะแคงโดยไม่มีใครรู้
    ไฟล์ที่ preprocess ไว้ก่อนมี meta.json ไม่มีข้อมูลนี้ (คืนว่าไม่ทราบ ไม่ใช่ว่าปกติ)."""
    from ..stages import read_meta

    out = []
    for stem, page in page_images(paths.preprocess_final(dataset, profile)).items():
        if stem in stems and (err := read_meta(page).get("orientation_error")):
            out.append((stem, str(err)[:90]))
    return out


def stems_needing(check, stems: list[str]) -> list[str]:
    """ไฟล์ใน ``stems`` ที่ ``check(stem)`` เป็นเท็จ (ต้องทำใหม่)."""
    return [s for s in stems if not check(s)]


def final_stems(paths: DataPaths, dataset: str, only: list[str] | None, profile: str = DEFAULT_PROFILE) -> list[str]:
    final_dir: Path = paths.preprocess_final(dataset, profile)
    stems = list(page_images(final_dir)) if final_dir.is_dir() else []
    return [s for s in stems if not only or any(s.endswith(o) for o in only)]
