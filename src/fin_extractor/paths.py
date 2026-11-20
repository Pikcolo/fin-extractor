"""ตำแหน่งเก็บข้อมูลทั้งหมดของโปรเจกต์ (ที่เดียว) — โค้ดและสคริปต์ต้องถามจากที่นี่ ไม่ฝังพาธเอง.

โฟลเดอร์ตั้งชื่อตามลำดับขั้น เปิดดูไล่ตามเลขได้เลย (ทุกอย่างใต้ ``paths.data_dir`` ใน config ค่าตั้งต้น ``data/``
และ ``.gitignore`` ไม่ให้ commit อะไรใต้นี้ เพราะมาจากเอกสารจริง) ทุกขั้นใช้ชื่อชุดข้อมูล ``<dataset>`` และชื่อไฟล์ ``<ไฟล์>`` เหมือนกัน
ไฟล์เดียวจึงตามดูข้ามขั้นได้ด้วยชื่อเดียว

    01_input/<dataset>/                          ไฟล์อินพุต (.pdf .jpg .png)
    02_preprocess/<dataset>/<profile>/           preprocess ชุดตั้งค่าหนึ่ง (profile)
        steps/        ภาพทีละขั้น (เมื่อใส่ --steps)
        final/        ภาพสุดท้ายที่ส่งต่อให้ทุกขั้นถัดไป
    03_sections/<dataset>/<profile>/             แบ่งหน้าเป็นส่วน (section) จาก final/ ไว้ดู/debug
        layout/<ไฟล์>.json         ผังการแบ่ง (bbox, ชนิด, keep) + ลักษณะภาพ ·  <ไฟล์>_layout.png ภาพตรวจด้วยตา
        images/<วิธี>/<ไฟล์>/      ภาพส่วนที่เตรียมก่อน OCR: sNN.png + index.json  (วิธี = luma | flat | bin)
        regions/<ไฟล์>/            โหมดแบ่ง region/ตาราง (คำสั่ง regions): ภาพแต่ละขั้น, crops/ (region), crops/tables/ (ช่อง), manifest.json
    04_ocr/<dataset>/<engine>/                   ข้อความจาก OCR ของตัวอ่านหนึ่งตัว (paddle, typhoon, openthai, oxr)
        <วิธี>/<ไฟล์>/   ``page`` = อ่านทั้งหน้า · ``luma|flat|bin|color`` = อ่านทีละ section (มี .sections.json .md .chunks.json)
        visualize/       ภาพให้ดูว่าอ่านอย่างไร
    05_extract/<dataset>/<engine>/<วิธี>/<extractor>/<ไฟล์>_p<หน้า>.json   ผลตาม schema extractor = rules | llm | hybrid
    etc/benchmarks/  etc/compare/                งานทดสอบ/เปรียบเทียบที่ข้าม engine/ข้ามไฟล์
    etc/experiments/<ชื่อ>/                      การทดลองครั้งเดียวที่ไม่ใช่ส่วนของ pipeline
    reports/                                     รายงานภาพ

ถ้ารันหนึ่งครั้งมีไฟล์อินพุตมากกว่า ``NEST_OVER`` (2) ไฟล์ ผลของแต่ละไฟล์จะอยู่ในโฟลเดอร์ย่อยชื่อไฟล์ ถ้าไม่เกินก็วางตรง ๆ
ผู้เขียนใช้ ``doc_dir`` ผู้อ่านใช้ ``find_dir`` (รับได้ทั้งสองแบบ) ชื่อไฟล์ผลใช้ ``<ไฟล์>_p<หน้า>`` เหมือนกันทุกขั้น
ลบผลของ engine หรือวิธีที่ไม่ใช้ = ลบโฟลเดอร์ ``<engine>``/``<วิธี>`` ใน 04_ocr และ 05_extract
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .config import resolve_path

FONT_DIR = "assets/fonts"
MODEL_DIR = "models"

DEFAULT_FOLDERS = {
    "raw": "01_input",
    "preprocess": "02_preprocess",
    "sections": "03_sections",
    "ocr": "04_ocr",
    "extract": "05_extract",
    "benchmarks": "etc/benchmarks",
    "compare": "etc/compare",
    "experiments": "etc/experiments",
    "reports": "reports",
}
DEFAULT_PROFILE = "default"
NEST_OVER = 2  # รันที่มีไฟล์อินพุตเกินจำนวนนี้ → โฟลเดอร์ย่อยต่อไฟล์

# ชื่อโฟลเดอร์ย่อยที่เขียนซ้ำหลายที่ — ให้ที่เดียว (โค้ดอื่นห้ามเขียนสตริงเหล่านี้เอง)
FINAL_DIR = "final"  # ภาพ preprocess สุดท้ายใต้ 02_preprocess/<dataset>/<profile>/
STEPS_DIR = "steps"  # ภาพ preprocess ทีละขั้น
LAYOUT_DIR = "layout"  # ผังแบ่ง section ใต้ 03_sections/<dataset>/<profile>/
IMAGES_DIR = "images"  # ภาพส่วนที่เตรียมก่อน OCR (images/<วิธี>/<ไฟล์>/)
REGIONS_DIR = "regions"  # โหมดแบ่ง region/ตาราง (regions/<ไฟล์>/ ภาพแต่ละขั้น + crops/ + manifest.json)
PAGE = "page"  # "วิธี" ของผลที่อ่านทั้งหน้า (ไม่แบ่ง section) ใต้ 04_ocr/…/<engine>/ และ 05_extract/…/<engine>/


def variant_of(variant: str | None) -> str:
    """ชื่อโฟลเดอร์ระดับ <วิธี>: อ่านทั้งหน้า = ``page`` นอกนั้นคือวิธีเตรียมภาพของ section (luma | flat | bin | color)."""
    return variant or PAGE


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

    def preprocess_final(self, dataset: str, profile: str = DEFAULT_PROFILE) -> Path:
        return self.preprocess(dataset, profile) / FINAL_DIR

    # ผลของตัวอ่าน (04_ocr) และผลสกัด (05_extract) — โครงเดียวกัน <dataset>/<engine>/<วิธี>/
    def ocr_engine(self, engine: str, dataset: str) -> Path:
        return self._top("ocr") / slug(dataset) / slug(engine)

    def raw_data(self, engine: str, dataset: str, variant: str | None = None) -> Path:
        """ข้อความดิบจาก OCR: ``04_ocr/<dataset>/<engine>/<วิธี>/`` (วิธี = page เมื่ออ่านทั้งหน้า)."""
        return self.ocr_engine(engine, dataset) / variant_of(variant)

    def final(self, engine: str, dataset: str, extractor: str, variant: str | None = None) -> Path:
        """ผลตาม schema: ``05_extract/<dataset>/<engine>/<วิธี>/<extractor>/``."""
        return self._top("extract") / slug(dataset) / slug(engine) / variant_of(variant) / slug(extractor)

    def visualize(self, engine: str, dataset: str) -> Path:
        return self.ocr_engine(engine, dataset) / "visualize"

    # ข้ามengine / ข้ามไฟล์
    def benchmarks(self) -> Path:
        return self._top("benchmarks")

    def compare(self, name: str | None = None) -> Path:
        return self._top("compare") / slug(name) if name else self._top("compare")

    # ผลขั้นแบ่ง section: 03_sections/<dataset>/<profile>/ (ตัดจากภาพ final/ ของ profile นั้น)
    def sections(self, dataset: str, profile: str = DEFAULT_PROFILE) -> Path:
        return self._top("sections") / slug(dataset) / slug(profile)

    def section_layout(self, dataset: str, profile: str = DEFAULT_PROFILE) -> Path:
        """ผังแบ่ง section: ``<ไฟล์>.json`` (ลักษณะภาพ + ``layout_sections``) และ ``<ไฟล์>_layout.png`` ภาพตรวจด้วยตา"""
        return self.sections(dataset, profile) / LAYOUT_DIR

    def section_images(self, dataset: str, stem: str | None = None, variant: str | None = None, profile: str = DEFAULT_PROFILE) -> Path:
        """ภาพส่วนที่เตรียมก่อน OCR ``images/<วิธี>/<ไฟล์>/`` (สร้างด้วย scripts/sections/make_clean_sections.py) วิธี = luma | flat | bin"""
        base = self.sections(dataset, profile) / IMAGES_DIR
        if variant:
            base = base / variant
        return base / slug(stem) if stem else base

    def section_regions(self, dataset: str, stem: str | None = None, profile: str = DEFAULT_PROFILE) -> Path:
        """โหมดแบ่ง region/ตาราง ``regions/<ไฟล์>/`` (ดู sections/regions/run.py) ไม่มี ``stem`` = โฟลเดอร์รวม (summary.json, contact_sheet.png)"""
        base = self.sections(dataset, profile) / REGIONS_DIR
        return base / slug(stem) if stem else base

    # ชื่อเดิม (ใช้ในสคริปต์/notebook ที่ยังไม่ย้าย)
    section_features = section_layout
    section_clean = section_images

    def experiments(self, name: str | None = None) -> Path:
        return self._top("experiments") / slug(name) if name else self._top("experiments")

    def reports(self, name: str | None = None) -> Path:
        return self._top("reports") / slug(name) if name else self._top("reports")
