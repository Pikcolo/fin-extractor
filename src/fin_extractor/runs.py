"""โครงเก็บผลของทุกคำสั่งให้เหมือนกัน.

    data/output/<คำสั่ง>/<ชื่ออินพุต>__<แท็ก>/
        manifest.json       คำสั่งที่รัน, ค่าที่ใช้, เวลา, รายการไฟล์พร้อมสถานะ
        results/ | text/ | images/ | debug/   ผลของคำสั่งนั้น

- ``extract``    → ``results/<ไฟล์>_p<หน้า>.json`` (ตาม ``schemas/invoice.schema.json``), ``debug/`` เมื่อใส่ ``--debug``
- ``ocr``        → ``text/<ไฟล์>_p<หน้า>.txt`` (หรือ ``.md`` เมื่อ engine เป็น vlm)
- ``preprocess`` → ``images/<ไฟล์>_p<หน้า>.png``, ``steps/`` เมื่อใส่ ``--steps``

แท็กบอกวิธีที่ใช้ เช่น ``ocr-rules``, ``vlm-llm`` รันซ้ำด้วยอินพุตและแท็กเดิมจะเขียนทับโฟลเดอร์เดิม
ใส่ ``--out`` เพื่อกำหนดโฟลเดอร์เองทั้งโฟลเดอร์ (ยังได้โครงข้างในแบบเดียวกัน)
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from . import __version__
from .config import resolve_path
from .schema import SCHEMA_VERSION


def _slug(text: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", text, flags=re.UNICODE).strip("_") or "input"


class Run:
    """โฟลเดอร์ผลของการรันหนึ่งครั้ง + manifest ที่เขียนตอนจบ (หรือตอนล้ม)."""

    def __init__(self, command: str, input_path: str | Path, tag: str | None, cfg: dict[str, Any],
                 out: str | Path | None = None, settings: dict[str, Any] | None = None):
        input_path = Path(input_path)
        if out:
            self.dir = Path(out)
        else:
            name = _slug(input_path.stem if input_path.is_file() else input_path.name)
            name = f"{name}__{_slug(tag)}" if tag else name
            self.dir = resolve_path(cfg["paths"]["output_dir"]) / command / name
        self.command = command
        self.input = str(input_path)
        self.settings = settings or {}
        self.preprocessing = cfg.get("preprocessing")
        self.files: list[dict[str, Any]] = []
        self.started = datetime.now().astimezone()

    def sub(self, name: str) -> Path:
        """โฟลเดอร์ย่อยของรันนี้ (สร้างให้)."""
        p = self.dir / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def add(self, **entry: Any) -> None:
        self.files.append(entry)

    def write_manifest(self, status: str = "completed") -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "command": self.command,
            "status": status,
            "argv": sys.argv[1:],
            "input": self.input,
            "started_at": self.started.isoformat(timespec="seconds"),
            "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "fin_extractor_version": __version__,
            "schema_version": SCHEMA_VERSION,
            "settings": self.settings,
            "preprocessing": self.preprocessing,
            "files": self.files,
        }
        path = self.dir / "manifest.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return path

    def __enter__(self) -> Run:
        self.dir.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.write_manifest("completed" if exc_type is None else "failed")
