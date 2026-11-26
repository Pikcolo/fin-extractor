"""Orchestrator: ข้อความที่อ่านได้ → extract → validate → ExtractionResult (ตัวอ่านทุก engine ใช้ทางเดียวกัน ดู readers/).

คำสั่ง ``extract`` ใช้ ``process_raw`` (สกัดจากข้อความรายส่วนที่เก็บไว้ ไม่สร้างตัวอ่าน ไม่อ่านภาพ)
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Literal

from .doctypes import (
    GENERIC_PROFILE,
    Profile,
    classify,
    core_labels_for,
    load_core_labels,
    load_profiles,
    squash,
)
from .extraction import extract_fields, lines_with_chunks, markdown_to_lines
from .extraction.rule_extractor import SUPPLEMENT, chunk_lines
from .extraction.core_labels import extract_party_name
from .extraction.hybrid import known_context, merge_llm, top_level, unresolved_keys
from .extraction.llm_extractor import profile_schema
from .extraction.profile_extractor import (
    dedupe_remark,
    extract_document_fields,
    extract_remark,
    extract_unmapped,
)
from .readers import reader_label
from .readers.lines import normalize_thai
from .schema import ExtractionResult, InvoiceData, RejectedValue, ValidationCheck
from .validation import decide_status, run_checks
from .validation.checks import NOT_APPLICABLE_DETAIL, income_rows_sum
from .validation.evidence import build_field_evidence, comparable, field_grounding, flatten, review_from_evidence

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
    strategy: str = "winner"  # ค่าตั้งต้นระดับ class (พฤติกรรมเดิม) ``__init__`` ตั้งทับตาม config / --llm hybrid
    last_rejected: list = []

    def __init__(
        self,
        cfg: dict[str, Any],
        engine: str = "openthai",
        use_llm: bool = False,
        strategy: Literal["winner", "hybrid"] | None = None,
    ):
        """สกัดจากข้อความที่อ่านไว้แล้วเท่านั้น (ไม่สร้างตัวอ่าน ไม่แตะ GPU) ``engine`` ใช้แค่ตั้งชื่อในผล."""
        self.cfg = cfg
        self.use_llm = use_llm
        # winner = rules กับ LLM สกัดแยกกัน แล้วเลือกชุดที่ check ดีกว่า | hybrid = rules นำ LLM เติมเฉพาะช่องที่ยังไม่เสร็จ (extraction/hybrid.py)
        self.strategy = strategy or cfg.get("extraction", {}).get("strategy", "winner")
        if self.strategy not in ("winner", "hybrid"):
            raise ValueError(f"unknown extraction strategy: {self.strategy}")
        self.last_rejected: list[tuple[str, str]] = []  # ค่าที่ LLM เสนอแต่ด่านของ hybrid ปัดตก (คีย์, เหตุผล) ของหน้าล่าสุด

        self.reader = SimpleNamespace(name=reader_label(engine, cfg))  # ชื่อเดียวกับตัวอ่านจริง ให้ engine_name ตรงกับตอนอ่าน
        self.profiles = load_profiles(cfg.get("document_types") or {})
        self.core_labels = load_core_labels(cfg.get("core_labels"))

        self.llm = None
        if self.use_llm:
            from .extraction.llm_extractor import LLMExtractor

            self.llm = LLMExtractor(cfg["llm_extractor"])

    @property
    def engine_name(self) -> str:
        if self.llm and self.strategy == "hybrid":
            return f"{self.reader.name}+rules+{self.llm.name}"
        return self.reader.name + ("+" + self.llm.name if self.llm else "+rules")

    @staticmethod
    def _incomplete_check() -> ValidationCheck:
        return ValidationCheck(name="reader_complete", passed=False, detail="ตัวอ่านตอบไม่จบ (ชนเพดาน token) — ข้อมูลอาจขาด")

    @staticmethod
    def _chunks_of(lines: list[str], raw_text: str) -> list[str | None] | None:
        """รหัส section ของแต่ละบรรทัด (ขนานกับ ``lines``) จากคอมเมนต์ ``<!-- sNN -->`` ใน Markdown ของขั้น OCR รายส่วน
        None ถ้าไม่มีคอมเมนต์ หรือบรรทัดไม่ตรงกับ ``lines`` (ไม่เดา)."""
        if "<!-- s" not in raw_text:
            return None
        pairs = lines_with_chunks(raw_text)
        return [c for _ln, c in pairs] if [ln for ln, _c in pairs] == lines else None

    @staticmethod
    def lines_from_raw(raw_text: str) -> list[str]:
        """ข้อความดิบที่เก็บไว้ → บรรทัดสำหรับขั้นสกัด (แบบเดียวกับตอนอ่านสด ทุก engine) ข้อความรายส่วนตัดหัวตารางที่อ่านซ้ำออก (lines_with_chunks)."""
        if "<!-- s" in raw_text:
            return [ln for ln, _c in lines_with_chunks(raw_text)]
        return markdown_to_lines(raw_text)

    def _consistency(self, d: InvoiceData) -> float:
        checks = run_checks(d, self.cfg["validation"]["amount_tolerance"],
                            self.cfg["validation"]["default_vat_rate"])
        return sum(c.passed is True for c in checks) - sum(c.passed is False for c in checks)

    def profile_for(self, data: InvoiceData) -> Profile:
        """โปรไฟล์ชนิดเอกสารที่ใช้สกัดผลนี้ (ชื่ออยู่ใน ``data.document_profile``)."""
        return next((p for p in self.profiles if p.name == data.document_profile), GENERIC_PROFILE)

    def extract_traced(
        self, lines: list[str], raw_text: str
    ) -> tuple[InvoiceData, dict[str, str], dict[str, bool | None]]:
        """``_extract`` + ที่มาของค่าที่พบเฉพาะในท่อนเสริมของชั้นตรวจความครบ (ตัวอ่านหลักอ่านส่วนนั้นไม่ครบ) = ชื่อตัวอ่านเสริม."""
        data, sources, agree = self._extract(lines, raw_text)
        if SUPPLEMENT not in raw_text:
            return data, sources, agree
        rows = chunk_lines(raw_text)
        engines = sorted({s for _ln, _c, s in rows if s})
        if not engines:
            return data, sources, agree
        profile = self.profile_for(data)
        main = field_grounding(data, "\n".join(ln for ln, _c, s in rows if not s), profile)
        extra = field_grounding(data, "\n".join(ln for ln, _c, s in rows if s), profile)
        for key, src in sources.items():
            if src == "rules" and main.get(key) is False and extra.get(key):
                sources[key] = engines[0]
        return data, sources, agree

    def _extract(
        self, lines: list[str], raw_text: str
    ) -> tuple[InvoiceData, dict[str, str], dict[str, bool | None]]:
        """rules ทำงานเสมอ; ถ้ามี LLM ด้วยจะเลือกผลที่ตัวเลขสอดคล้องกันมากกว่า แล้วเติม field ที่ขาดจากอีกฝั่ง.

        LLM เล็ก ๆ (เช่น 3B) มักสลับ field ยอดเงิน ส่วน rules เปราะกับ layout — ใช้ cross-check เป็นกรรมการ

        คืน (ข้อมูล, ต้นทางของแต่ละ field, ผลเทียบสองแหล่ง) คีย์แบบ ``flatten`` — ต้นทางคือ "rules" หรือ "llm"
        ผลเทียบเป็น True/False เมื่อทั้งสองแหล่งมีค่า ไม่มีคีย์ = มีแหล่งเดียว (รายการสินค้าไม่เทียบรายตัว
        เพราะสองแหล่งแบ่งรายการไม่ตรงกัน ใช้ check items_sum ยืนยันแทน)
        """
        rules = extract_fields(lines, raw_text)
        profile = classify(self.profiles, "\n".join(lines), rules.document_type)
        rules.document_profile = profile.name
        rules.document_fields = extract_document_fields(lines, profile)
        labels = core_labels_for(profile, self.core_labels)
        for party, key in ((rules.buyer, "buyer.name"), (rules.vendor, "vendor.name")):
            name, branch = extract_party_name(lines, labels, key)
            if name and not party.name:
                party.name = name
                party.branch = party.branch or branch
        chunks = self._chunks_of(lines, raw_text)
        rules.remark = [*extract_remark(lines, profile, rules, chunks), *extract_unmapped(lines, rules, chunks)]
        if not self.llm:
            return rules, dict.fromkeys(flatten(rules), "rules"), {}
        if self.strategy == "hybrid":
            return self._extract_hybrid(rules, profile, lines, raw_text)
        try:
            llm = self.llm.extract(normalize_thai(raw_text), profile)  # เหมือน hybrid: LLM ลอก "ํา" (นิคหิต+า) ของ OpenThai มา ค่าจึงไม่ตรงกับบรรทัดที่ normalize แล้ว (ถูกนับว่าไม่พบในข้อความ)
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

    def _score(self, d: InvoiceData, profile: Profile) -> float:
        checks = run_checks(d, self.cfg["validation"]["amount_tolerance"],
                            self.cfg["validation"]["default_vat_rate"], applicable=profile.checks, words=profile.words)
        return sum(c.passed is True for c in checks) - sum(c.passed is False for c in checks)

    def _extract_hybrid(self, rules: InvoiceData, profile: Profile, lines: list[str], raw_text: str):
        """rules นำ → หาช่องที่ยังไม่เสร็จ → ถาม LLM เฉพาะช่องนั้น → รับค่าที่ผ่านด่าน (ดู extraction/hybrid.py)."""
        self.last_rejected = []
        sources = dict.fromkeys(flatten(rules), "rules")
        na = set(profile.not_applicable)
        checks = run_checks(rules, self.cfg["validation"]["amount_tolerance"],
                            self.cfg["validation"]["default_vat_rate"], applicable=profile.checks, words=profile.words)
        unresolved = unresolved_keys(rules, checks, profile.required, na, profile.primary_amount)
        asked = {top_level(k) for k in unresolved}
        if not profile_schema(profile, asked)["properties"]:  # ไม่มีช่องไหนที่ LLM ตอบได้ (เช่น ชนิดนี้ไม่มีช่องนั้น)
            return rules, sources, {}
        try:
            llm = self.llm.extract(normalize_thai(raw_text), profile, only=asked, known=known_context(rules, unresolved),
                                    focus=unresolved)
        except Exception as e:  # noqa: BLE001 — LLM ล่ม/ตอบผิด schema → ใช้ rules
            log.warning("LLM extraction failed, falling back to rules: %s", e)
            return rules, sources, {}
        merged, accepted, self.last_rejected, agree = merge_llm(
            rules, llm, asked, unresolved, "\n".join(lines), profile, lambda d: self._score(d, profile))
        merged.document_profile = profile.name
        merged.remark = dedupe_remark(merged.remark, merged, profile)
        for key in flatten(merged):
            sources[key] = "llm" if key in accepted else sources.get(key, "rules")
        return merged, sources, agree

    def assess(self, data: InvoiceData, lines: list[str], sources: dict[str, str] | None = None,
               agree: dict[str, bool | None] | None = None, extra: list[ValidationCheck] | tuple = ()):
        """ตรวจข้อมูลที่สกัดแล้ว (หรือที่คนแก้แล้ว) → (checks, ฟิลด์ที่ขาด, สถานะ, หลักฐานรายฟิลด์) ไม่สกัดใหม่ ใช้โปรไฟล์ตาม ``data.document_profile``
        ``lines`` = บรรทัดของข้อความที่อ่านได้ (ใช้ตรวจว่าค่าปรากฏในข้อความไหม)."""
        profile = self.profile_for(data)
        checks = run_checks(data, self.cfg["validation"]["amount_tolerance"],
                            self.cfg["validation"]["default_vat_rate"], applicable=profile.checks, words=profile.words)
        # 50 ทวิ: ยอดรวม = ผลรวมแถวรายการ ต้องใช้บรรทัดข้อความ (แถวรายการไม่ได้เก็บใน data) ชนิดอื่นรายงานว่าไม่ใช้ ให้ทุกผลมีชื่อ check ชุดเดียวกัน
        if profile.checks is not None and "income_rows_sum" in profile.checks:
            checks.append(income_rows_sum(lines, data, tol=self.cfg["validation"]["amount_tolerance"]))
        else:
            checks.append(ValidationCheck(name="income_rows_sum", passed=None, detail=NOT_APPLICABLE_DETAIL))
        checks += list(extra)
        na = set(profile.not_applicable)
        missing, status = decide_status(data, checks, not_applicable=na, required=profile.required,
                                        primary_amount=profile.primary_amount,
                                        can_auto_approve=profile.can_auto_approve)
        evidence = build_field_evidence(data, checks, "\n".join(lines), sources, agree, not_applicable=na,
                                        required=profile.required, profile=profile)
        return checks, missing, status, evidence

    def build_result(self, source_file: str, page: int, lines: list[str], raw_text: str,
                     ocr_conf: float | None, extra: list[ValidationCheck]) -> ExtractionResult:
        """ข้อความที่อ่านได้ของหน้าหนึ่ง → ผลสกัด/ตรวจ/สถานะ (ไม่แตะ OCR จึงรันซ้ำจาก ``raw_text`` ที่เก็บไว้ได้)."""
        data, sources, agree = self.extract_traced(lines, raw_text)
        checks, _missing, status, evidence = self.assess(data, lines, sources, agree, extra)
        if status == "auto_approved" and any(src not in ("rules", "llm", "human") and key in evidence and evidence[key].level != "verified"
                                             for key, src in sources.items()):
            # ค่าจากตัวอ่านเสริมที่ไม่มี check ยืนยัน ต้องให้คนดู: ตัวอ่านเสริมอ่านวันที่/เลขที่เพี้ยนได้ (1318: 16/10 แทน 15/10, ST69100-0036)
            status = "needs_review"
        review = review_from_evidence(evidence, checks, self.profile_for(data).primary_amount)
        return ExtractionResult(
            source_file=source_file, page=page, engine=self.engine_name, data=data,
            checks=checks, review=review, ocr_confidence=round(ocr_conf, 4) if ocr_conf is not None else None,
            status=status, raw_text=raw_text,
            rejected=[RejectedValue(key=k, reason=why) for k, why in (self.last_rejected if self.strategy == "hybrid" else [])],
        )

    def process_raw(self, source_file: str, page_no: int, raw_text: str, meta: dict | None = None) -> ExtractionResult:
        """สกัดซ้ำจากข้อความดิบที่เก็บไว้ (ไม่เรียก OCR) ``meta`` = ไฟล์ .meta.json ของข้อความนั้น."""
        meta = meta or {}
        extra = [self._incomplete_check()] if meta.get("truncated") else []
        return self.build_result(source_file, page_no, self.lines_from_raw(raw_text), raw_text,
                                 meta.get("ocr_confidence"), extra)
