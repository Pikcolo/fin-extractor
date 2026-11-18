"""ตัดภาพส่วนจากภาพ final ตามผังที่เก็บไว้ (ขั้น 3 → ใช้ในขั้น 4)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ..io import read_image
from ..paths import DEFAULT_PROFILE, DataPaths
from .layout import crop_full, resize_to_work


def load_final_image(paths: DataPaths, dataset: str, stem: str, profile: str = DEFAULT_PROFILE) -> np.ndarray | None:
    """ภาพทั้งหน้าที่ preprocess แล้ว (หน้าแรกของไฟล์ ``stem``) หรือ None ถ้ายังไม่มี."""
    for p in paths.preprocess_final(dataset, profile).rglob(f"{stem}_p1.png"):
        return read_image(p)
    return None


def layout_path(paths: DataPaths, dataset: str, stem: str, profile: str = DEFAULT_PROFILE) -> Path:
    return paths.section_layout(dataset, profile) / f"{stem}.json"


def iter_section_crops(paths: DataPaths, stem: str, dataset: str = "finance_doc_scan", keep_only: bool = True) -> Iterator[tuple[dict, np.ndarray]]:
    """(section, ภาพ BGR ความละเอียดเต็ม) ของทุกส่วนตามผังที่เก็บไว้ ตัดจากภาพทั้งหน้าที่ pipeline preprocess เก็บไว้ (ตรงแล้ว สี)."""
    full = load_final_image(paths, dataset, stem)
    if full is None:
        raise FileNotFoundError(f"ไม่พบภาพที่ preprocess แล้วของ {stem} ใน {paths.preprocess_final(dataset)}")
    work, _ = resize_to_work(full)
    layout = json.loads(layout_path(paths, dataset, stem).read_text(encoding="utf-8"))["layout_sections"]
    for s in layout:
        if keep_only and not s["keep"]:
            continue
        yield s, crop_full(full, work, s)
