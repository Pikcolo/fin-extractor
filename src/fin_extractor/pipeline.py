"""Orchestrator: ไฟล์ → หน้า → preprocess → OCR/VLM → extract → validate → ExtractionResult.

mode
- ``ocr`` : PaddleOCR + rules (หรือ + LLM ถ้า use_llm)
- ``vlm`` : Typhoon OCR อ่านทั้งหน้าเป็น Markdown + LLM (หรือ + rules ถ้า use_llm=False)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from .extraction import extract_fields, markdown_to_lines
from .io import load_pages
from .ocr import create_engine
from .preprocessing import preprocess
from .schema import ExtractionResult, InvoiceData, ValidationCheck
from .validation import decide_status, run_checks
from .validation.scoring import not_applicable_for
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
    ):
        self.cfg = cfg
        self.mode = mode
        self.use_llm = (mode == "vlm") if use_llm is None else use_llm

        if mode == "ocr":
            self.reader = create_engine(engine, cfg)
        elif mode == "vlm":
            from .vlm import TyphoonOCR

            self.reader = TyphoonOCR(cfg["vlm"])
        else:
            raise ValueError(f"unknown mode: {mode}")
        self.profile = cfg["preprocessing"]

        self.llm = None
        if self.use_llm:
            from .extraction.llm_extractor import LLMExtractor

            self.llm = LLMExtractor(cfg["llm_extractor"])

    @property
    def engine_name(self) -> str:
        return self.reader.name + ("+" + self.llm.name if self.llm else "+rules")

    def read_page(self, img, debug_dir: Path | None = None, stem: str = "page"):
        """คืน (lines, raw_text, ocr_confidence, extra_checks)."""
        pre = preprocess(img, self.profile, keep_steps=debug_dir is not None)
        if debug_dir is not None:
            pre.save_steps(debug_dir, stem)
        log.info("preprocess %s: %s", stem, pre.meta)

        res = self.reader.recognize(pre.image)
        if self.mode == "ocr":
            return [ln.text for ln in res.lines], res.text, res.mean_confidence, []

        extra = []
        if res.truncated:
            extra.append(ValidationCheck(name="vlm_complete", passed=False,
                                         detail="VLM ตอบไม่จบ (max_tokens) — ข้อมูลอาจขาด"))
        return markdown_to_lines(res.markdown), res.markdown, None, extra

    def _consistency(self, d: InvoiceData) -> float:
        checks = run_checks(d, self.cfg["validation"]["amount_tolerance"],
                            self.cfg["validation"]["default_vat_rate"])
        return sum(c.passed is True for c in checks) - sum(c.passed is False for c in checks)

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
        if not self.llm:
            return rules, dict.fromkeys(flatten(rules), "rules"), {}
        try:
            llm = self.llm.extract(raw_text)
        except Exception as e:  # noqa: BLE001 — LLM ล่ม/ตอบผิด schema → ใช้ rules
            log.warning("LLM extraction failed, falling back to rules: %s", e)
            return rules, dict.fromkeys(flatten(rules), "rules"), {}
        if self._consistency(llm) >= self._consistency(rules):
            (best, best_name), (other, other_name) = (llm, "llm"), (rules, "rules")
        else:
            (best, best_name), (other, other_name) = (rules, "rules"), (llm, "llm")
        log.info("extraction winner: %s", best_name)
        merged = fill_missing(best, other)

        flat_best, flat_other = flatten(best), flatten(other)
        sources: dict[str, str] = {}
        agree: dict[str, bool | None] = {}
        for key in flatten(merged):
            is_list = key.startswith(("items[", "extra_fields["))  # สองแหล่งแบ่งรายการไม่ตรงกัน เทียบรายตัวไม่ได้
            list_name = key.split("[", 1)[0]
            sources[key] = best_name if (getattr(best, list_name) if is_list else key in flat_best) else other_name
            if not is_list and key in flat_best and key in flat_other:
                agree[key] = comparable(flat_best[key]) == comparable(flat_other[key])
        return merged, sources, agree

    def process_file(self, path: str | Path, debug_dir: str | Path | None = None) -> list[ExtractionResult]:
        path = Path(path)
        pages = load_pages(path, self.cfg["loader"]["pdf_dpi"])
        results = []
        for page_no, img in enumerate(pages, start=1):
            stem = f"{path.stem}_p{page_no}"
            lines, raw_text, ocr_conf, extra = self.read_page(
                img, Path(debug_dir) if debug_dir else None, stem
            )
            data, sources, agree = self.extract_traced(lines, raw_text)
            checks = run_checks(data, self.cfg["validation"]["amount_tolerance"],
                                self.cfg["validation"]["default_vat_rate"]) + extra
            na = not_applicable_for(data.document_type, self.cfg.get("document_types"))
            missing, status = decide_status(data, checks, not_applicable=na)
            evidence = build_field_evidence(data, checks, "\n".join(lines), sources, agree, not_applicable=na)
            results.append(ExtractionResult(
                source_file=str(path), page=page_no, engine=self.engine_name, data=data,
                checks=checks, field_evidence=evidence, ocr_confidence=round(ocr_conf, 4) if ocr_conf is not None else None,
                missing_fields=missing, status=status, raw_text=raw_text,
            ))
        return results
