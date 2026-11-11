"""ขั้นที่ใช้ร่วมกันของคำสั่ง preprocess / ocr / extract.

1. ``prepared_pages``: หน้าที่ผ่าน preprocess แล้ว เก็บภาพสุดท้ายไว้ที่ ``preprocess/<dataset>/<profile>/final/``
   ทุก engine ใช้ภาพชุดเดียวกัน (ไม่ preprocess ซ้ำ) ถ้าตั้งค่า preprocess เปลี่ยนจากที่เก็บไว้ ต้องใช้ ``--profile`` ใหม่
   หรือ ``--force`` (ล้างโฟลเดอร์ profile นั้นแล้วทำใหม่) กันภาพคนละตั้งค่าปนกันใน profile เดียว
2. ``save_raw`` / ``load_raw``: ข้อความดิบจาก OCR ที่ ``result_ocr/<engine>/<dataset>/raw_data/<ไฟล์>_p<หน้า>.(txt|md)``
   พร้อม ``.meta.json`` (ความมั่นใจ, ตอบไม่จบหรือไม่, ตั้งค่า profile) ใช้สกัดซ้ำด้วย rules หรือ LLM โดยไม่เรียก OCR อีก
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .io import load_pages, read_image, save_image
from .io.loader import page_count
from .preprocessing import preprocess

log = logging.getLogger(__name__)
SETTINGS_FILE = "settings.json"


def preprocess_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    """ค่าที่กำหนดภาพสุดท้าย (เทียบว่า profile ที่เก็บไว้ใช้ได้ไหม)."""
    return json.loads(json.dumps({"preprocessing": cfg["preprocessing"], "pdf_dpi": cfg["loader"]["pdf_dpi"]}, default=str))


def prepared_pages(path: str | Path, cfg: dict[str, Any], profile_dir: Path, *, steps: bool = False,
                   force: bool = False) -> Iterator[tuple[int, np.ndarray, dict[str, Any]]]:
    """(เลขหน้า, ภาพที่ผ่าน preprocess, meta) ของไฟล์ — ใช้ภาพที่เก็บไว้ถ้าตั้งค่าตรงกันและครบทุกหน้า."""
    path = Path(path)
    final = profile_dir / "final"
    wanted = preprocess_settings(cfg)
    settings_path = profile_dir / SETTINGS_FILE
    stored = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else None

    cached = [final / f"{path.stem}_p{i}.png" for i in range(1, page_count(path) + 1)]
    if stored == wanted and not force and not steps and all(p.exists() for p in cached):
        for i, p in enumerate(cached, start=1):
            yield i, read_image(p), {"cached": True}
        return

    if stored is not None and stored != wanted:
        if not force:
            raise ValueError(f"{profile_dir} ถูกสร้างด้วยตั้งค่า preprocess อื่น — ใช้ --profile ชื่อใหม่ "
                             "หรือ --force เพื่อล้างโฟลเดอร์ profile นี้แล้วทำใหม่")
        log.warning("ล้าง %s แล้วทำ preprocess ใหม่ (ตั้งค่าเปลี่ยน)", profile_dir)
        shutil.rmtree(final, ignore_errors=True)
        shutil.rmtree(profile_dir / "steps", ignore_errors=True)
    profile_dir.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(wanted, ensure_ascii=False, indent=2), encoding="utf-8")

    for i, img in enumerate(load_pages(path, cfg["loader"]["pdf_dpi"]), start=1):
        res = preprocess(img, cfg["preprocessing"], keep_steps=steps)
        save_image(final / f"{path.stem}_p{i}.png", res.image)
        if steps:
            res.save_steps(profile_dir / "steps", f"{path.stem}_p{i}")
        yield i, res.image, res.meta


def save_raw(raw_dir: Path, stem: str, text: str, ext: str, meta: dict[str, Any]) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_dir / f"{stem}.{ext}"
    out.write_text(text, encoding="utf-8")
    (raw_dir / f"{stem}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return out


def raw_pages(raw_dir: Path, file_stem: str, ext: str) -> list[tuple[int, Path]]:
    """(เลขหน้า, ไฟล์ข้อความดิบ) ของไฟล์หนึ่ง เรียงตามหน้า."""
    pages = []
    for p in raw_dir.glob(f"{file_stem}_p*.{ext}"):
        m = re.fullmatch(re.escape(file_stem) + r"_p(\d+)", p.stem)
        if m:
            pages.append((int(m.group(1)), p))
    return sorted(pages)


def load_raw(path: Path) -> tuple[str, dict[str, Any]]:
    """(ข้อความดิบ, meta) — meta ว่างถ้าไม่มีไฟล์ ``.meta.json``."""
    meta_path = path.with_name(path.stem + ".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return path.read_text(encoding="utf-8"), meta
