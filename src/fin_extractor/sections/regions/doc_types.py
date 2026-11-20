"""ชนิดเอกสารจากตาราง configs/regions/doc_types.yaml (lookup ตามชื่อไฟล์ ไม่ใช่ตัวจำแนก)

คัดลอกจาก learning/preprocess_finance (src/finance_ocr/layout/classify.py) ตรงตัว ค่าคงที่วัดบนสแกนชุดเดียวกัน 18 ใบที่ 200 dpi — ใช้กับภาพที่ย่อเป็น 200 dpi เท่านั้น (ดู run.py)

Decide which document type (and therefore which region template) a page uses.

There is no OCR at this stage, so classification is a lookup in configs/doc_types.yaml
(exact doc_id -> type, or glob pattern -> type) with a "generic" fallback.
Replace `classify()` with a keyword/visual classifier later without touching other stages.
"""
from __future__ import annotations

from fnmatch import fnmatch


def classify(doc_id: str, doc_types: dict) -> tuple[str, dict]:
    """Return (doc_type, per-document overrides)."""
    docs = doc_types.get("documents", {}) or {}
    entry = docs.get(doc_id)
    if entry is None:
        for pattern, e in (doc_types.get("patterns", {}) or {}).items():
            if fnmatch(doc_id, pattern):
                entry = e
                break
    if entry is None:
        return doc_types.get("default_type", "generic"), {}
    if isinstance(entry, str):
        return entry, {}
    entry = dict(entry)
    return entry.pop("type", doc_types.get("default_type", "generic")), entry
