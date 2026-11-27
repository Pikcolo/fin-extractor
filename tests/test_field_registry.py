"""รายชื่อฟิลด์แกนกลางอยู่ที่ schema.py ที่เดียว: เทสต์นี้ล้มเมื่อเพิ่ม/เปลี่ยนฟิลด์ในโมเดลแล้วที่อื่นที่ต้องรู้จักไม่ตามมา (บอกว่าที่ไหน)."""

import ast
import json
import re
from decimal import Decimal
from pathlib import Path

from fin_extractor import groundtruth
from fin_extractor.doctypes import PARTY_ATTR_LABELS
from fin_extractor.extraction import hybrid, llm_extractor
from fin_extractor.extraction.rule_extractor import AMOUNT_FIELDS
from fin_extractor.schema import CORE_AMOUNT_KEYS, ITEM_KEYS, MONEY_KEYS, PARTY_KEYS, InvoiceData, LineItem, Party
from fin_extractor.validation import evidence

ROOT = Path(__file__).resolve().parents[1]


def decimal_fields(model) -> set[str]:
    return {name for name, f in model.model_fields.items() if "Decimal" in str(f.annotation)}


def test_registry_matches_the_models():
    assert set(CORE_AMOUNT_KEYS) == decimal_fields(InvoiceData)  # ฟิลด์ Decimal ใหม่ใน InvoiceData ต้องเข้า CORE_AMOUNT_KEYS (ยอดเงินหรืออัตรา)
    assert "vat_rate" not in MONEY_KEYS and set(MONEY_KEYS) | {"vat_rate"} == set(CORE_AMOUNT_KEYS)
    assert PARTY_KEYS == tuple(Party.model_fields) and ITEM_KEYS == tuple(LineItem.model_fields)


def test_every_consumer_knows_every_amount_key():
    assert set(CORE_AMOUNT_KEYS) <= set(groundtruth.FIELDS)
    assert set(CORE_AMOUNT_KEYS) <= groundtruth.MONEY
    assert set(CORE_AMOUNT_KEYS) <= hybrid.AMOUNT_KEYS
    assert set(CORE_AMOUNT_KEYS) <= set(llm_extractor._AMOUNT_KEYS)
    assert set(CORE_AMOUNT_KEYS) <= set(evidence._SCALARS)
    assert set(MONEY_KEYS) == evidence._AMOUNT_KEYS


def test_amount_labels_of_the_rules_name_real_fields():
    assert {name for name, _rx in AMOUNT_FIELDS} <= set(CORE_AMOUNT_KEYS)


def test_party_label_table_covers_each_party_field():
    assert tuple(PARTY_ATTR_LABELS) == PARTY_KEYS


def test_published_json_schema_lists_every_amount_key():
    schema = json.loads((ROOT / "schemas" / "invoice.schema.json").read_text(encoding="utf-8"))
    props = None

    def find(o):
        nonlocal props
        if isinstance(o, dict):
            if "amount_before_vat" in o and isinstance(o["amount_before_vat"], dict):
                props = o
                return
            for v in o.values():
                find(v)
        elif isinstance(o, list):
            for v in o:
                find(v)

    find(schema)
    assert props is not None and set(CORE_AMOUNT_KEYS) <= set(props)


def groundtruth_labels() -> dict:
    tree = ast.parse((ROOT / "scripts" / "eval" / "groundtruth_app.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "LABELS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("ไม่พบ LABELS ใน groundtruth_app.py")


def test_ground_truth_form_has_a_label_for_every_core_field():
    labels = groundtruth_labels()
    missing = [k for k in groundtruth.FIELDS if k not in labels]
    assert not missing, f"groundtruth_app.LABELS ขาดป้ายของ {missing}"


def test_demo_page_has_a_label_for_every_core_field():
    html = (ROOT / "scripts" / "demo" / "ocr_demo.html").read_text(encoding="utf-8")
    block = html[html.index("const FIELD_LABEL"):]
    block = block[:block.index("};")]
    keys = set(re.findall(r'"([A-Za-z_.]+)":', block))
    missing = [k for k in groundtruth.FIELDS if k not in keys]
    assert not missing, f"ocr_demo.html FIELD_LABEL ขาดป้ายของ {missing}"


def test_amount_keys_are_decimals_in_the_models():
    d = InvoiceData(**{k: Decimal("1.00") for k in MONEY_KEYS})
    assert all(getattr(d, k) == Decimal("1.00") for k in MONEY_KEYS)
