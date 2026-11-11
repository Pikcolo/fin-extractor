"""manifest ของการรันหนึ่งครั้ง — เขียนไว้ในโฟลเดอร์ผลของรันนั้น.

โครงโฟลเดอร์ทั้งหมดอยู่ที่ ``paths.py`` (preprocess/<dataset>/<profile>/, result_ocr/<engine>/<dataset>/{raw_data,final/<extractor>}/)
``Run`` รับโฟลเดอร์จากผู้เรียก แล้วเขียน ``manifest.json`` ตอนจบ (หรือตอนล้ม) ไว้ข้างผลของมัน:

    preprocess  → preprocess/<dataset>/<profile>/manifest.json            (ภาพอยู่ใน final/, steps/)
    ocr         → result_ocr/<engine>/<dataset>/raw_data/manifest.json
    extract     → result_ocr/<engine>/<dataset>/final/<extractor>/manifest.json

manifest เก็บ: คำสั่งที่รัน, ค่าที่ใช้, เวลา, รายการไฟล์พร้อมสถานะ และพาธผลแบบสัมพัทธ์กับโฟลเดอร์ของ manifest
รันซ้ำด้วยโฟลเดอร์เดิมจะเขียนทับ
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from . import __version__
from .schema import SCHEMA_VERSION


class Run:
    """โฟลเดอร์ผลของการรันหนึ่งครั้ง + manifest ที่เขียนตอนจบ (หรือตอนล้ม)."""

    def __init__(self, command: str, input_path: str | Path, cfg: dict[str, Any], out_dir: str | Path,
                 settings: dict[str, Any] | None = None):
        self.dir = Path(out_dir)
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
