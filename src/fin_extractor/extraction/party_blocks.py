"""อ่านผู้ขาย/ผู้ซื้อจาก "บล็อกของแต่ละฝ่าย" ที่แบบฟอร์มพิมพ์ป้ายแยกไว้ (ตอนนี้: 50 ทวิ ``ผู้มีหน้าที่หักภาษี`` / ``ผู้ถูกหักภาษี``).

ฟอร์มแบบนี้บอกฝ่ายด้วยป้ายของบล็อก ไม่ใช่ตำแหน่ง (บริษัทเดียวกันเป็นผู้หักในเอกสารหนึ่งและเป็นผู้ถูกหักในอีกใบได้) กฎตามตำแหน่งของ
``rule_extractor`` ("ก่อนเจอลูกค้า = ผู้ขาย") จึงใช้กับชนิดนี้ไม่ได้ ป้ายและฝั่งที่ตรงกันมาจาก config ``document_types.<ชนิด>.party_blocks``

วิธีอ่าน:
1. บล็อกเริ่มที่บรรทัดที่ **ขึ้นต้นด้วย** ป้ายของฝ่ายนั้น (บรรทัดคำแนะนำสำเนา "ฉบับที่ 1 (สำหรับผู้ถูกหักภาษี ...)" ไม่ขึ้นต้นด้วยป้าย จึงไม่นับ)
   สิ้นสุดที่บล็อกของอีกฝ่าย หรือบรรทัดที่มีคำสิ้นสุด (``end``: หัวตารางเงินได้) ป้ายที่เจอซ้ำใช้อันแรก
2. เลขภาษี = เลข 13 หลักแรกในบล็อกที่ผ่าน checksum (OCR ทำเลขหลุดเป็น 12 หลัก = ไม่ผ่าน = ว่าง ไม่เดา)
3. ชื่อ = ค่าหลังป้าย "ชื่อ" ในบล็อก (ใช้ตัวอ่านเดียวกับ core_labels: ตัดคำหยุด/คำกำกับ ลองบรรทัดถัดไปเมื่อป้ายอยู่ลำพัง) ข้ามบรรทัดที่ขึ้นต้นด้วย "("
   เพราะเป็นคำแนะนำของฟอร์ม ("(ให้ระบุ ชื่ออาคาร/...") แล้วย้าย "(สำนักงานใหญ่)" ท้ายชื่อไปช่องสาขา
ที่อยู่ไม่อ่านที่นี่ (``assign_addresses`` ผูกที่อยู่กับฝ่ายที่ชื่ออยู่เหนือมัน ใช้ได้กับค่าที่ตั้งไว้แล้วเหมือนเดิม)
"""

from __future__ import annotations

import re

from ..doctypes import FieldSpec, PartyBlocks, alias_regex
from ..schema import InvoiceData, Party
from ..validation.tax_id import is_valid_thai_tax_id
from .core_labels import _read, split_branch
from .patterns import find_tax_ids

# ป้ายชื่อในบล็อก: "ชื่อ" · OCR สะกดเพี้ยน "ซื่อ" "ชือซี"
_NAME_SPEC = FieldSpec(key="buyer.name", label="ชื่อ", aliases=("ชื่อ", "ซื่อ", "ชือซี"), next_line=True)
_TRAILING_LEADERS = re.compile(r"\s*(?:\.{2,}|…)[\s.…]*$")  # "นายสมชาย ใจดี ..." จุดไข่ปลาที่ฟอร์มพิมพ์เป็นเส้นให้เขียน
_HEAD_OFFICE = re.compile(r"สำนักงานใหญ่|head\s*office", re.I)


def _starts(lines: list[str], blocks: PartyBlocks) -> dict[str, int]:
    """ดัชนีบรรทัดเริ่มบล็อกของแต่ละฝ่าย (ป้ายอยู่ต้นบรรทัด ใช้อันแรก)."""
    out: dict[str, int] = {}
    for party, aliases in blocks.parties.items():
        rx = re.compile(r"^\s*(?:" + "|".join(alias_regex(a).pattern for a in aliases) + ")", re.I)
        for i, line in enumerate(lines):
            if rx.match(line):
                out[party] = i
                break
    return out


def split_blocks(lines: list[str], blocks: PartyBlocks) -> dict[str, list[str]]:
    """{ฝ่าย: บรรทัดของบล็อกนั้น} เฉพาะฝ่ายที่เจอป้าย."""
    starts = _starts(lines, blocks)
    end_rx = re.compile("|".join(alias_regex(a).pattern for a in blocks.end), re.I) if blocks.end else None
    out: dict[str, list[str]] = {}
    for party, i in starts.items():
        stops = [j for j in starts.values() if j > i]
        if end_rx:
            stops += [j for j in range(i + 1, len(lines)) if end_rx.search(lines[j])][:1]
        out[party] = lines[i:min(stops) if stops else len(lines)]
    return out


def _read_party(block: list[str]) -> Party:
    party = Party()
    for line in block:
        for tid in find_tax_ids(line):
            if party.tax_id is None and is_valid_thai_tax_id(tid):
                party.tax_id = tid
    body = [ln for ln in block if not ln.lstrip().startswith("(")]  # รวมบรรทัดป้ายบล็อก (ชื่อติดป้ายบรรทัดเดียวกันได้)
    if (value := _read(_NAME_SPEC, body)):
        name, branch = split_branch(_TRAILING_LEADERS.sub("", value))
        party.name, party.branch = name or None, branch
    if party.branch is None and any(_HEAD_OFFICE.search(ln) for ln in block):
        party.branch = "สำนักงานใหญ่"
    return party


def fill_party_blocks(lines: list[str], data: InvoiceData, blocks: PartyBlocks) -> None:
    """ตั้ง ``data.vendor`` / ``data.buyer`` จากบล็อกของฝ่ายนั้นใหม่ทั้งก้อน (ไม่ใช่เติมช่องว่าง: ค่าที่ rules ตามตำแหน่งใส่ไว้อาจเป็นของอีกฝ่าย
    เช่น กรณีที่เลขภาษีของผู้ถูกหักไปอยู่ในช่องผู้หัก) ฝ่ายที่ไม่เจอป้ายก็ว่าง ไม่คงค่าตามตำแหน่งไว้."""
    found = split_blocks(lines, blocks)
    for party in blocks.parties:
        setattr(data, party, _read_party(found[party]) if party in found else Party())
