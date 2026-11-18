"""flow หลักห้าขั้น (``python -m fin_extractor sections``): ตัดสินว่าขั้นไหนของแต่ละไฟล์ "ยังใช้ได้" จึงข้ามได้

    preprocess → layout → images → ocr → extract
    ภาพ final     ผัง      ภาพรายส่วน  ข้อความ+ที่มา   JSON ตาม schema

กติกาข้ามของแต่ละขั้น (เทียบกับผลของขั้นก่อนหน้าเท่านั้น ไม่เดา)
- layout: มี ``layout/<ไฟล์>.json`` ที่พารามิเตอร์ (table_mode, max_piece_frac) ตรงกับที่สั่ง และใหม่กว่าภาพ final
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
                      profile: str = DEFAULT_PROFILE) -> bool:
    path = paths.section_layout(dataset, profile) / f"{stem}.json"
    page = page_images(paths.preprocess_final(dataset, profile)).get(stem)
    if not path.exists() or page is None or path.stat().st_mtime < page.stat().st_mtime:
        return False
    params = json.loads(path.read_text(encoding="utf-8")).get("layout_params", {})
    return params.get("table_mode") == table_mode and params.get("max_piece_frac") == max_piece_frac


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


def stems_needing(check, stems: list[str]) -> list[str]:
    """ไฟล์ใน ``stems`` ที่ ``check(stem)`` เป็นเท็จ (ต้องทำใหม่)."""
    return [s for s in stems if not check(s)]


def final_stems(paths: DataPaths, dataset: str, only: list[str] | None, profile: str = DEFAULT_PROFILE) -> list[str]:
    final_dir: Path = paths.preprocess_final(dataset, profile)
    stems = list(page_images(final_dir)) if final_dir.is_dir() else []
    return [s for s in stems if not only or any(s.endswith(o) for o in only)]
