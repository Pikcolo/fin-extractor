"""ตำแหน่งเก็บข้อมูลทั้งหมดของโปรเจกต์ (ที่เดียว) — โค้ดและสคริปต์ต้องถามจากที่นี่ ไม่ฝังพาธเอง.

โครง (ทุกอย่างใต้ ``paths.data_dir`` ใน config ค่าตั้งต้น ``data/`` และ ``.gitignore`` ไม่ให้ commit อะไรใต้นี้เลย เพราะมาจากเอกสารจริง)

    raw/<dataset>/                              ไฟล์อินพุต
    preprocess/<dataset>/<profile>/             ผลของ preprocess ชุดตั้งค่าหนึ่ง (profile)
        steps/        ภาพทีละขั้น (เมื่อใส่ --steps)
        final/        ภาพสุดท้ายที่ส่งต่อให้ OCR/VLM ทุกตัว
    result_ocr/<engine>/<dataset>/              ผลของตัวอ่านหนึ่งตัว (paddle, typhoon, oxr)
        raw_data/     ข้อความดิบจาก OCR (.txt/.md) เขียนครั้งเดียวต่อ engine
        final/<extractor>/   ผลตาม schema (.json) extractor = rules | llm
        visualize/    ภาพให้ดูว่าอ่านอย่างไร
    etc/benchmarks/  etc/compare/               งานทดสอบ/เปรียบเทียบที่ข้าม engine/ข้ามไฟล์
    reports/                                    รายงานภาพ

ถ้ารันหนึ่งครั้งมีไฟล์อินพุตมากกว่า ``NEST_OVER`` (2) ไฟล์ ผลของแต่ละไฟล์ (preprocess/final/steps, raw_data, final/<extractor>)
จะอยู่ในโฟลเดอร์ย่อยชื่อไฟล์ เช่น ``raw_data/001312/001312_p1.txt`` เพื่อดูทีละไฟล์ง่าย ถ้าไม่เกินก็วางตรง ๆ ใต้โฟลเดอร์นั้น
ผู้เขียนใช้ ``doc_dir`` ผู้อ่านใช้ ``find_dir`` (รับได้ทั้งสองแบบ จึงอ่านผลที่ปนกันได้)

``<dataset>`` คือชื่อโฟลเดอร์อินพุต (ไฟล์เดี่ยวใช้ชื่อโฟลเดอร์ที่ไฟล์อยู่) ชื่อไฟล์ผลใช้ ``<ไฟล์>_p<หน้า>`` เหมือนกันทุกขั้น
จึงดูไฟล์เดียวข้ามขั้น/ข้าม engine ได้ด้วยชื่อเดียว ลบ engine หรือ extractor ที่ไม่ใช้ = ลบโฟลเดอร์เดียว
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .config import resolve_path

FONT_DIR = "assets/fonts"
MODEL_DIR = "models"

DEFAULT_FOLDERS = {
    "raw": "raw",
    "preprocess": "preprocess",
    "result_ocr": "result_ocr",
    "benchmarks": "etc/benchmarks",
    "compare": "etc/compare",
    "reports": "reports",
}
DEFAULT_PROFILE = "default"
NEST_OVER = 2  # รันที่มีไฟล์อินพุตเกินจำนวนนี้ → โฟลเดอร์ย่อยต่อไฟล์


def slug(text: str) -> str:
    """ชื่อที่ใช้เป็นโฟลเดอร์ได้ (ตัดตัวคั่นพาธ/อักขระแปลก ๆ ออก)."""
    # ตัด "." หัวท้ายด้วย ไม่งั้นชื่อ ".." ไต่ขึ้นโฟลเดอร์แม่ได้
    return re.sub(r"[^\w.\-]+", "_", str(text), flags=re.UNICODE).strip("_.") or "x"


def dataset_of(input_path: str | Path) -> str:
    """ชื่อชุดข้อมูลของอินพุต: โฟลเดอร์ = ชื่อโฟลเดอร์, ไฟล์ = ชื่อโฟลเดอร์ที่ไฟล์อยู่."""
    p = Path(input_path)
    return slug(p.name if p.is_dir() else p.resolve().parent.name)


def nest_per_file(n_inputs: int) -> bool:
    return n_inputs > NEST_OVER


def doc_dir(base: Path, stem: str, nest: bool) -> Path:
    """โฟลเดอร์ที่ผู้เขียนใช้เก็บผลของไฟล์ ``stem`` ใต้ ``base`` (ถ้าเคยมีโฟลเดอร์ย่อยอยู่แล้วก็ใช้ต่อ กันผลกระจายสองที่)."""
    sub = base / slug(stem)
    return sub if nest or sub.is_dir() else base


def find_dir(base: Path, stem: str) -> Path:
    """โฟลเดอร์ที่ผู้อ่านควรมองหาผลของไฟล์ ``stem`` (โฟลเดอร์ย่อยถ้ามี ไม่งั้นตรง ๆ ใต้ ``base``)."""
    sub = base / slug(stem)
    return sub if sub.is_dir() else base


def iter_results(base: Path, pattern: str) -> list[Path]:
    """ไฟล์ที่ตรง ``pattern`` ใต้ ``base`` ทั้งแบบตรง ๆ และในโฟลเดอร์ย่อยต่อไฟล์ (ไม่รวม manifest.json)."""
    return sorted(p for p in [*base.glob(pattern), *base.glob(f"*/{pattern}")] if p.name != "manifest.json")


def font_path(name: str = "Sarabun-Regular.ttf") -> Path:
    """ฟอนต์ที่มากับโปรเจกต์ (``assets/fonts/``) ไม่พึ่งฟอนต์ของเครื่อง."""
    return resolve_path(FONT_DIR) / name


def model_dir(name: str = "") -> Path:
    """ที่เก็บโมเดลที่ library ดาวน์โหลดเอง (``models/`` ในโปรเจกต์ ไม่ commit)."""
    root = resolve_path(MODEL_DIR)
    return root / name if name else root


class DataPaths:
    """ผู้ให้พาธของทุกขั้น (ไม่สร้างโฟลเดอร์เอง ผู้เขียนไฟล์สร้างเมื่อจะเขียน)."""

    def __init__(self, cfg: dict[str, Any] | None = None):
        paths = (cfg or {}).get("paths", {})
        self.root = resolve_path(paths.get("data_dir", "data"))
        self._names = {**DEFAULT_FOLDERS, **paths.get("folders", {})}

    def _top(self, key: str) -> Path:
        return self.root / self._names[key]

    # อินพุต / preprocess
    def raw(self, dataset: str | None = None) -> Path:
        return self._top("raw") / slug(dataset) if dataset else self._top("raw")

    def preprocess(self, dataset: str, profile: str = DEFAULT_PROFILE) -> Path:
        return self._top("preprocess") / slug(dataset) / slug(profile)

    def preprocess_steps(self, dataset: str, profile: str = DEFAULT_PROFILE) -> Path:
        return self.preprocess(dataset, profile) / "steps"

    def preprocess_final(self, dataset: str, profile: str = DEFAULT_PROFILE) -> Path:
        return self.preprocess(dataset, profile) / "final"

    # ผลของตัวอ่าน
    def engine(self, engine: str, dataset: str) -> Path:
        return self._top("result_ocr") / slug(engine) / slug(dataset)

    def raw_data(self, engine: str, dataset: str) -> Path:
        return self.engine(engine, dataset) / "raw_data"

    def final(self, engine: str, dataset: str, extractor: str) -> Path:
        return self.engine(engine, dataset) / "final" / slug(extractor)

    def visualize(self, engine: str, dataset: str) -> Path:
        return self.engine(engine, dataset) / "visualize"

    # ข้ามengine / ข้ามไฟล์
    def benchmarks(self) -> Path:
        return self._top("benchmarks")

    def compare(self, name: str | None = None) -> Path:
        return self._top("compare") / slug(name) if name else self._top("compare")

    def reports(self, name: str | None = None) -> Path:
        return self._top("reports") / slug(name) if name else self._top("reports")
