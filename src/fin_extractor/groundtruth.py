"""Ground truth ที่คนกรอกในเครื่อง (``scripts/eval/groundtruth_app.py``) และการให้คะแนนผลสกัดเทียบกับมัน

ไฟล์อยู่ที่ ``data/etc/groundtruth/<dataset>/<ไฟล์>.json`` (ใต้ data/ จึงไม่ถูก commit และไม่ออกนอกเครื่อง)

    {"source": "<ไฟล์>", "page": 1, "status": "draft" | "done", "prefill_from": "openthai/luma/rules", "updated": "...", "note": "",
     "fields": {"vendor.name": "...", "invoice_no": null, ...},      ค่าที่ถูกตามเอกสาร · null = เอกสารไม่มีช่องนี้ · ไม่มีคีย์ = ยังไม่ได้ตรวจ
     "document_fields": {"total_paid": "2000.00", ...},              ฟิลด์ประจำชนิดเอกสาร (กติกาเดียวกับ fields)
     "items": [{"description": "...", "quantity": "1", "unit_price": "2000.00", "amount": "2000.00"}], "items_checked": true}

คะแนน (``score``) เทียบทีละช่องหลัง normalize ตามชนิดค่า (เงิน = ค่าทศนิยม, วันที่ = ISO, เลขภาษี = ตัวเลข, ข้อความ = ตัดช่องว่าง + แก้สระอำ)
ไม่มีเกณฑ์ความคล้าย: ตรงหรือไม่ตรงเท่านั้น ข้อความที่ต่างกันแค่ตัวเดียวก็นับว่าไม่ตรง
"""

from __future__ import annotations

import json
import re
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .paths import DataPaths, find_dir
from .readers.lines import normalize_thai

PARTY = ("name", "tax_id", "branch", "address")
FIELDS = (  # ช่องแกนกลางที่กรอก/ให้คะแนน (ลำดับ = ลำดับในฟอร์ม) document_type/profile/remark ไม่อยู่ที่นี่: เป็นการตัดสินของระบบ ไม่ใช่ค่าบนเอกสาร
    *(f"vendor.{k}" for k in PARTY), *(f"buyer.{k}" for k in PARTY),
    "book_no", "invoice_no", "date",
    "subtotal", "discount", "amount_before_vat", "non_taxable_amount", "vat_rate", "vat_amount", "total_amount",
)
MONEY = {"subtotal", "discount", "amount_before_vat", "non_taxable_amount", "vat_amount", "total_amount", "quantity", "unit_price", "amount", "vat_rate"}
ITEM_KEYS = ("description", "quantity", "unit_price", "amount")
OUTCOMES = ("correct", "wrong", "missed", "spurious", "both_empty")


def gt_dir(paths: DataPaths, dataset: str) -> Path:
    return paths.groundtruth(dataset)


def gt_path(paths: DataPaths, dataset: str, stem: str) -> Path:
    return gt_dir(paths, dataset) / f"{stem}.json"


def load_gt(paths: DataPaths, dataset: str, stem: str) -> dict | None:
    p = gt_path(paths, dataset, stem)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def save_gt(paths: DataPaths, dataset: str, stem: str, doc: dict) -> Path:
    p = gt_path(paths, dataset, stem)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def result_path(paths: DataPaths, dataset: str, run: str, stem: str, page: int = 1) -> Path:
    """ผลสกัดของ ``run`` = ``<engine>/<วิธี>/<extractor>`` เช่น ``openthai/luma/rules``."""
    engine, variant, extractor = run.split("/")
    base = paths.final(engine, dataset, extractor, variant)
    return find_dir(base, stem) / f"{stem}_p{page}.json"


def load_result(paths: DataPaths, dataset: str, run: str, stem: str) -> dict | None:
    p = result_path(paths, dataset, run, stem)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def result_values(result: dict) -> tuple[dict[str, Any], dict[str, Any], list[dict]]:
    """(ช่องแกนกลาง, document_fields, items) จาก JSON ผลสกัด (``ExtractionResult``) ค่าเป็นสตริงหรือ None."""
    data = result.get("data") or {}

    def get(key: str):
        cur: Any = data
        for part in key.split("."):
            cur = cur.get(part) if isinstance(cur, dict) else None
        return None if cur in (None, "") else str(cur)

    items = [{k: (None if it.get(k) in (None, "") else str(it.get(k))) for k in ITEM_KEYS} for it in data.get("items") or []]
    return {k: get(k) for k in FIELDS}, {k: str(v) for k, v in (data.get("document_fields") or {}).items()}, items


def prefill(result: dict | None, run: str) -> dict:
    """ฉบับร่างจากผลสกัด: คนแค่ยืนยันหรือแก้ (ยังไม่มีคีย์ไหนถือว่าตรวจแล้วจนกว่าจะบันทึก)."""
    fields, doc_fields, items = result_values(result) if result else ({k: None for k in FIELDS}, {}, [])
    return {"status": "new", "prefill_from": run if result else None, "fields": fields, "document_fields": doc_fields,
            "items": items, "items_checked": False, "note": ""}


# --------------------------------------------------------------------------- เทียบค่า

def _money(v: str) -> Decimal | None:
    s = re.sub(r"[,\s฿]|บาท|THB", "", v)
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def kind_of(key: str, doc_types: dict[str, str] | None = None) -> str:
    """ชนิดค่าของช่อง: money | date | tax_id | text ฟิลด์ประจำชนิด (``document_fields.<k>``) ใช้ชนิดที่โปรไฟล์ประกาศ (``doc_types``)."""
    leaf = key.rsplit(".", 1)[-1]
    if key.startswith("document_fields."):
        t = (doc_types or {}).get(leaf, "text")
        return "money" if t in ("money", "percent") else t
    return "money" if leaf in MONEY else leaf if leaf in ("date", "tax_id") else "text"


def normalize(key: str, value: Any, kind: str | None = None) -> Any:
    """รูปที่ใช้เทียบ: None/ว่าง = None · เงิน/จำนวน = Decimal · วันที่ = ISO · เลขภาษี = ตัวเลขล้วน · ข้อความ = ตัดช่องว่าง + แก้สระอำ."""
    if value is None or str(value).strip() == "":
        return None
    v = str(value).strip()
    leaf = kind or kind_of(key)
    if leaf == "money":
        m = _money(v)
        if m is not None:
            return m.normalize()
    if leaf == "date":
        try:
            return date.fromisoformat(v).isoformat()
        except ValueError:
            pass
    if leaf == "tax_id":
        return re.sub(r"\D", "", v)
    return "".join(normalize_thai(v).split())


def outcome(truth: Any, got: Any, key: str, kind: str | None = None) -> str:
    t, g = normalize(key, truth, kind), normalize(key, got, kind)
    if t is None and g is None:
        return "both_empty"
    if t is None:
        return "spurious"
    if g is None:
        return "missed"
    return "correct" if t == g else "wrong"


def score(gt: dict, result: dict | None, doc_types: dict[str, str] | None = None) -> dict[str, str]:
    """{คีย์: ผล} เฉพาะช่องที่คนตรวจแล้ว (มีคีย์ใน ground truth) items นับเป็นช่อง ``items.count`` + ``items[i].<คีย์>``
    (เทียบตามลำดับ ใช้เมื่อ ``items_checked``) ``result`` None = ไม่มีผลสกัด ทุกช่องที่มีค่าเป็น missed
    ``doc_types`` = ชนิดของฟิลด์ประจำชนิดเอกสาร ({คีย์: money|percent|date|tax_id|text} จากโปรไฟล์ ``profile_types``)."""
    fields, doc_fields, items = result_values(result) if result else ({}, {}, [])
    out = {k: outcome(v, fields.get(k), k) for k, v in (gt.get("fields") or {}).items()}
    out.update({f"document_fields.{k}": outcome(v, doc_fields.get(k), f"document_fields.{k}", kind_of(f"document_fields.{k}", doc_types))
                for k, v in (gt.get("document_fields") or {}).items()})
    if gt.get("items_checked"):
        truth = gt.get("items") or []
        out["items.count"] = "correct" if len(truth) == len(items) else "wrong"
        for i, t in enumerate(truth):
            g = items[i] if i < len(items) else {}
            for k in ITEM_KEYS:
                out[f"items[{i}].{k}"] = outcome(t.get(k), g.get(k), k)
    return out


def profile_types(profiles, name: str | None) -> dict[str, str]:
    """{คีย์ฟิลด์ประจำชนิด: ชนิด} ของโปรไฟล์ชื่อ ``name`` (``doctypes.load_profiles``) ไม่เจอ = {} (เทียบเป็นข้อความ)."""
    p = next((p for p in profiles if p.name == name), None)
    return {f.key: f.type for f in p.fields} if p else {}


def summarize(outcomes: list[dict[str, str]]) -> dict[str, Any]:
    """นับผลรวมหลายไฟล์ ``accuracy`` = ถูก / ช่องที่เอกสารมีค่า (correct + wrong + missed) ``spurious`` = ระบบใส่ค่าในช่องที่เอกสารไม่มี."""
    n = {o: 0 for o in OUTCOMES}
    for one in outcomes:
        for v in one.values():
            n[v] += 1
    present = n["correct"] + n["wrong"] + n["missed"]
    return {**n, "files": len(outcomes), "present": present, "accuracy": round(n["correct"] / present, 4) if present else None}
