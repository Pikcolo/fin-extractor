"""Orchestrator: ไฟล์ → หน้า → preprocess → OCR/VLM → extract → validate → ExtractionResult.

mode
- ``ocr`` : PaddleOCR + rules (หรือ + LLM ถ้า use_llm)
- ``vlm`` : Typhoon OCR อ่านทั้งหน้าเป็น Markdown + LLM (หรือ + rules ถ้า use_llm=False)
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

from .doctypes import (
    GENERIC_PROFILE, Profile, classify, core_labels_for, legacy_not_applicable, load_core_labels, load_profiles, squash)
from .extraction.core_labels import extract_buyer_name
from .extraction import extract_fields, markdown_to_lines
from .extraction.profile_extractor import dedupe_remark, extract_document_fields, extract_remark
from .io import load_pages
from .ocr import create_engine
from .preprocessing import preprocess
from .schema import ExtractionResult, InvoiceData, ValidationCheck
from .validation import decide_status, run_checks
from .validation.evidence import build_field_evidence, comparable, flatten

log = logging.getLogger(__name__)


def fill_missing(primary: InvoiceData, secondary: InvoiceData) -> InvoiceData:
    """เติม field ที่เป็น None/ว่างใน ``primary`` ด้วยค่าจาก ``secondary`` (ไม่ทับค่าที่มีอยู่)."""
    merged = primary.model_copy(deep=True)
    for name in InvoiceData.model_fields:
        cur, alt = getattr(merged, name), getattr(secondary, name)
        if name in ("vendor", "buyer"):
            for sub in type(cur).model_fields:
                if getattr(cur, sub) in (None, ""):
                    setattr(cur, sub, getattr(alt, sub))
        elif name == "document_fields":
            cur.update({k: v for k, v in alt.items() if k not in cur})
        elif name == "remark":  # สองแหล่งเก็บหมายเหตุคนละชุด รวมกัน (ของผู้ชนะก่อน) ไม่ใช่ใช้ฝั่งเดียว
            have = {(squash(e.label), squash(e.value)) for e in cur}
            cur.extend(e for e in alt if (squash(e.label), squash(e.value)) not in have)
        elif cur in (None, "", []):
            setattr(merged, name, alt)
    return merged


class FinanceOCRPipeline:
    def __init__(
        self,
        cfg: dict[str, Any],
        mode: Literal["ocr", "vlm"] = "ocr",
        engine: str = "paddle",
        use_llm: bool | None = None,
        with_reader: bool = True,
    ):
        """``with_reader=False``: ไม่โหลดตัวอ่าน (OCR/VLM) ใช้สกัดซ้ำจากข้อความดิบที่เก็บไว้ (process_raw) โดยไม่แตะ GPU."""
        self.cfg = cfg
        self.mode = mode
        self.use_llm = (mode == "vlm") if use_llm is None else use_llm

        if mode not in ("ocr", "vlm"):
            raise ValueError(f"unknown mode: {mode}")
        if not with_reader:  # ใช้แค่ชื่อให้ engine_name ตรงกับตอนอ่านจริง
            self.reader = SimpleNamespace(name=engine if mode == "ocr" else f"vlm:{cfg['vlm']['model']}")
        elif mode == "ocr":
            self.reader = create_engine(engine, cfg)
        else:
            from .vlm import TyphoonOCR

            self.reader = TyphoonOCR(cfg["vlm"])
        self.profile = cfg["preprocessing"]
        self.doc_table = cfg.get("document_types") or {}
        self.profiles = load_profiles(self.doc_table)
        self.core_labels = load_core_labels(cfg.get("core_labels"))

        self.llm = None
        if self.use_llm:
            from .extraction.llm_extractor import LLMExtractor

            self.llm = LLMExtractor(cfg["llm_extractor"])

    @property
    def engine_name(self) -> str:
        return self.reader.name + ("+" + self.llm.name if self.llm else "+rules")

    def read_prepared(self, img):
        """อ่านข้อความจากภาพที่ผ่าน preprocess แล้ว คืน (lines, raw_text, ocr_confidence, extra_checks)."""
        res = self.reader.recognize(img)
        if self.mode == "ocr":
            return [ln.text for ln in res.lines], res.text, res.mean_confidence, []

        extra = []
        if res.truncated:
            extra.append(self._incomplete_check())
        return markdown_to_lines(res.markdown), res.markdown, None, extra

    @staticmethod
    def _incomplete_check() -> ValidationCheck:
        return ValidationCheck(name="vlm_complete", passed=False, detail="VLM ตอบไม่จบ (max_tokens) — ข้อมูลอาจขาด")

    def lines_from_raw(self, raw_text: str) -> list[str]:
        """ข้อความดิบที่เก็บไว้ → บรรทัดสำหรับขั้นสกัด (แบบเดียวกับตอนอ่านสด)."""
        if self.mode == "vlm":
            return markdown_to_lines(raw_text)
        return [ln for ln in raw_text.split("\n") if ln.strip()]

    def _consistency(self, d: InvoiceData) -> float:
        checks = run_checks(d, self.cfg["validation"]["amount_tolerance"],
                            self.cfg["validation"]["default_vat_rate"])
        return sum(c.passed is True for c in checks) - sum(c.passed is False for c in checks)

    def profile_for(self, data: InvoiceData) -> Profile:
        """โปรไฟล์ชนิดเอกสารที่ใช้สกัดผลนี้ (ชื่ออยู่ใน ``data.document_profile``)."""
        return next((p for p in self.profiles if p.name == data.document_profile), GENERIC_PROFILE)

    def extract(self, lines: list[str], raw_text: str) -> InvoiceData:
        return self.extract_traced(lines, raw_text)[0]

    def extract_traced(
        self, lines: list[str], raw_text: str
    ) -> tuple[InvoiceData, dict[str, str], dict[str, bool | None]]:
        """rules ทำงานเสมอ; ถ้ามี LLM ด้วยจะเลือกผลที่ตัวเลขสอดคล้องกันมากกว่า แล้วเติม field ที่ขาดจากอีกฝั่ง.

        LLM เล็ก ๆ (เช่น 3B) มักสลับ field ยอดเงิน ส่วน rules เปราะกับ layout — ใช้ cross-check เป็นกรรมการ

        คืน (ข้อมูล, ต้นทางของแต่ละ field, ผลเทียบสองแหล่ง) คีย์แบบ ``flatten`` — ต้นทางคือ "rules" หรือ "llm"
        ผลเทียบเป็น True/False เมื่อทั้งสองแหล่งมีค่า ไม่มีคีย์ = มีแหล่งเดียว (รายการสินค้าไม่เทียบรายตัว
        เพราะสองแหล่งแบ่งรายการไม่ตรงกัน ใช้ check items_sum ยืนยันแทน)
        """
        rules = extract_fields(lines)
        profile = classify(self.profiles, "\n".join(lines), rules.document_type)
        rules.document_profile = profile.name
        rules.document_fields = extract_document_fields(lines, profile)
        name, branch = extract_buyer_name(lines, core_labels_for(profile, self.core_labels))
        if name and not rules.buyer.name:
            rules.buyer.name = name
            rules.buyer.branch = rules.buyer.branch or branch
        rules.remark = extract_remark(lines, profile, rules)
        if not self.llm:
            return rules, dict.fromkeys(flatten(rules), "rules"), {}
        try:
            llm = self.llm.extract(raw_text, profile)
        except Exception as e:  # noqa: BLE001 — LLM ล่ม/ตอบผิด schema → ใช้ rules
            log.warning("LLM extraction failed, falling back to rules: %s", e)
            return rules, dict.fromkeys(flatten(rules), "rules"), {}
        if self._consistency(llm) >= self._consistency(rules):
            (best, best_name), (other, other_name) = (llm, "llm"), (rules, "rules")
        else:
            (best, best_name), (other, other_name) = (rules, "rules"), (llm, "llm")
        log.info("extraction winner: %s", best_name)
        merged = fill_missing(best, other)
        merged.document_profile = profile.name
        merged.remark = dedupe_remark(merged.remark, merged, profile)

        flat_best, flat_other = flatten(best), flatten(other)
        best_remark = {(squash(e.label), squash(e.value)) for e in best.remark}
        sources: dict[str, str] = {}
        agree: dict[str, bool | None] = {}
        for key in flatten(merged):
            is_list = key.startswith(("items[", "remark["))  # สองแหล่งแบ่งรายการไม่ตรงกัน เทียบรายตัวไม่ได้
            list_name = key.split("[", 1)[0]
            if list_name == "remark":  # รายการหมายเหตุที่รวมแล้ว: ดูว่าแต่ละรายการมาจากฝั่งไหน
                e = merged.remark[int(key[len("remark["):-1])]
                sources[key] = best_name if (squash(e.label), squash(e.value)) in best_remark else other_name
                continue
            sources[key] = best_name if (getattr(best, list_name) if is_list else key in flat_best) else other_name
            if not is_list and key in flat_best and key in flat_other:
                agree[key] = comparable(flat_best[key]) == comparable(flat_other[key])
        return merged, sources, agree

    def build_result(self, source_file: str, page: int, lines: list[str], raw_text: str,
                     ocr_conf: float | None, extra: list[ValidationCheck]) -> ExtractionResult:
        """ข้อความที่อ่านได้ของหน้าหนึ่ง → ผลสกัด/ตรวจ/สถานะ (ไม่แตะ OCR จึงรันซ้ำจาก ``raw_text`` ที่เก็บไว้ได้)."""
        data, sources, agree = self.extract_traced(lines, raw_text)
        profile = self.profile_for(data)
        checks = run_checks(data, self.cfg["validation"]["amount_tolerance"],
                            self.cfg["validation"]["default_vat_rate"], applicable=profile.checks) + extra
        na = set(profile.not_applicable) | legacy_not_applicable(self.doc_table, data.document_type)
        missing, status = decide_status(data, checks, not_applicable=na, required=profile.required,
                                        primary_amount=profile.primary_amount,
                                        can_auto_approve=profile.can_auto_approve)
        evidence = build_field_evidence(data, checks, "\n".join(lines), sources, agree, not_applicable=na,
                                        required=profile.required, profile=profile)
        return ExtractionResult(
            source_file=source_file, page=page, engine=self.engine_name, data=data,
            checks=checks, field_evidence=evidence, ocr_confidence=round(ocr_conf, 4) if ocr_conf is not None else None,
            missing_fields=missing, status=status, raw_text=raw_text,
        )

    def process_page(self, source_file: str, page_no: int, img) -> ExtractionResult:
        """ภาพที่ผ่าน preprocess แล้ว → อ่าน → สกัด → ผลของหน้านั้น."""
        lines, raw_text, ocr_conf, extra = self.read_prepared(img)
        return self.build_result(source_file, page_no, lines, raw_text, ocr_conf, extra)

    def process_raw(self, source_file: str, page_no: int, raw_text: str, meta: dict | None = None) -> ExtractionResult:
        """สกัดซ้ำจากข้อความดิบที่เก็บไว้ (ไม่เรียก OCR) ``meta`` = ไฟล์ .meta.json ของข้อความนั้น."""
        meta = meta or {}
        extra = [self._incomplete_check()] if meta.get("truncated") else []
        return self.build_result(source_file, page_no, self.lines_from_raw(raw_text), raw_text,
                                 meta.get("ocr_confidence"), extra)

    def process_file(self, path: str | Path) -> list[ExtractionResult]:
        """ไฟล์ → preprocess ในหน่วยความจำ → ผล (ไม่เก็บภาพ ใช้ใน script วัดผล; คำสั่ง CLI ใช้ stages.prepared_pages)."""
        path = Path(path)
        return [self.process_page(str(path), page_no, preprocess(img, self.profile).image)
                for page_no, img in enumerate(load_pages(path, self.cfg["loader"]["pdf_dpi"]), start=1)]
