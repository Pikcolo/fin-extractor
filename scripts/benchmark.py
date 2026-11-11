"""Benchmark: เทียบ engine หลายแบบบนเอกสารหลายแพทเทิร์นที่มีเฉลย

    # 1) เตรียมชุดข้อมูล (โหลดจาก Invoice Extraction Benchmark + เอกสารไทยที่มีเฉลย)
    python scripts/benchmark.py prepare --per-cell 2

    # 2) รันทีละ engine (paddle+llm และ vlm ต้องเปิด Ollama)
    python scripts/benchmark.py run paddle
    python scripts/benchmark.py run paddle+llm
    python scripts/benchmark.py run vlm

    # 3) สรุปผลทุก engine ที่รันแล้ว
    python scripts/benchmark.py report

ชุดข้อมูลอยู่ที่ data/raw/benchmark/ (ไม่ commit) ผลอยู่ที่ data/benchmarks/ieb/

ที่มาของเอกสาร
    Invoice Extraction Benchmark (c) 2026 InvoiceParser Pro — CC BY 4.0
    https://huggingface.co/datasets/drew-ipp/invoice-extraction-benchmark
"""

from __future__ import annotations

import argparse
import collections
import difflib
import json
import logging
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "benchmark"
OUT = ROOT / "data" / "benchmarks" / "ieb"

IEB = "drew-ipp/invoice-extraction-benchmark"
IEB_RAW = f"https://huggingface.co/datasets/{IEB}/resolve/main/corpus/v1/"

# engine ที่รองรับ: ชื่อ → (mode, ocr engine, ใช้ LLM ไหม)
ENGINES = {
    "paddle": ("ocr", "paddle", False),
    "paddle+llm": ("ocr", "paddle", True),
    "vlm": ("vlm", "paddle", True),
}

FIELDS = ["vendor_name", "invoice_number", "invoice_date", "subtotal", "tax", "total"]
MONEY = {"subtotal", "tax", "total"}


# --------------------------------------------------------------- prepare


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def prepare(per_cell: int, thai_dir: Path | None) -> int:
    """สุ่มแบบแบ่งชั้น: ทุกคู่ (หมวด, ความยาก) เอามาไม่เกิน per_cell ใบ

    เลือกตามลำดับในไฟล์ metadata (ไม่สุ่มจริง) เพื่อให้ได้ชุดเดิมทุกครั้งที่รัน
    ตัวเลขจึงเทียบข้ามรอบได้
    """
    rows = [json.loads(x) for x in _get(IEB_RAW + "metadata.jsonl").decode().splitlines() if x.strip()]
    picked: dict[tuple, list] = collections.defaultdict(list)
    for r in rows:
        cell = (r["category"], r["difficulty"])
        if len(picked[cell]) < per_cell:
            picked[cell].append(r)

    DATA.mkdir(parents=True, exist_ok=True)
    n = 0
    for cell, items in sorted(picked.items()):
        for r in items:
            suffix = Path(r["file_name"]).suffix
            doc = DATA / f"{r['doc_id']}{suffix}"
            if not doc.exists():
                doc.write_bytes(_get(IEB_RAW + r["file_name"]))
            truth = {
                "source": IEB,
                "category": r["category"],
                "difficulty": r["difficulty"],
                "language": "en",
                "pages": r.get("pages"),
                "expected": r["expected"],
            }
            (DATA / f"{r['doc_id']}.json").write_text(
                json.dumps(truth, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            n += 1
    print(f"IEB: {n} ฉบับ จาก {len(picked)} คู่หมวด×ความยาก")

    if thai_dir:
        n += _add_thai(thai_dir)

    (DATA / "ATTRIBUTION.txt").write_text(
        "Invoice Extraction Benchmark (c) 2026 InvoiceParser Pro\n"
        f"https://huggingface.co/datasets/{IEB}\n"
        "Creative Commons Attribution 4.0 (CC BY 4.0)\n",
        encoding="utf-8",
    )
    print(f"รวม {n} ฉบับ ที่ {DATA}")
    return 0


def _add_thai(src: Path) -> int:
    """เอกสารไทยที่มีเฉลยแบบเก่า (vendor_name / subtotal / vat / total / line_items)"""
    n = 0
    for answer in sorted(src.glob("*.json")):
        old = json.loads(answer.read_text(encoding="utf-8"))
        if old.get("_source"):  # มาจาก IEB อยู่แล้ว ไม่ซ้ำ
            continue
        doc = next((p for p in src.glob(answer.stem + ".*") if p.suffix != ".json"), None)
        if doc is None:
            continue
        expected = {
            "vendor_name": old.get("vendor_name"),
            "invoice_number": old.get("invoice_no"),
            "invoice_date": old.get("invoice_date"),
            "subtotal": _num(old.get("subtotal")),
            "tax": _num(old.get("vat")),
            "total": _num(old.get("total")),
            "lines": [{"amount": _num(x.get("amount"))} for x in old.get("line_items") or []],
            "should_flag": False,
            "scored": FIELDS + ["line_items", "flag"],
        }
        (DATA / doc.name).write_bytes(doc.read_bytes())
        (DATA / f"{answer.stem}.json").write_text(
            json.dumps(
                {"source": "thai", "category": "thai", "difficulty": "real",
                 "language": "th", "expected": expected},
                ensure_ascii=False, indent=1,
            ),
            encoding="utf-8",
        )
        n += 1
    print(f"ภาษาไทย: {n} ฉบับ")
    return n


def _num(v):
    if v in (None, ""):
        return None
    return float(str(v).replace(",", ""))


# ------------------------------------------------------------------- run


def _merge_pages(results):
    """รวมผลหลายหน้าเป็นเอกสารเดียว: ช่องที่ว่างเติมจากหน้าถัดไป รายการสินค้าต่อกัน"""
    from fin_extractor.pipeline import fill_missing

    data = results[0].data
    for r in results[1:]:
        data = fill_missing(data, r.data)
    if len(results) > 1:
        data.items = [it for r in results for it in r.data.items]
    rank = {"failed": 2, "needs_review": 1, "auto_approved": 0}
    status = max((r.status for r in results), key=rank.__getitem__)
    return data, status


def run(name: str) -> int:
    from fin_extractor.config import load_config
    from fin_extractor.pipeline import FinanceOCRPipeline

    if name not in ENGINES:
        print(f"ไม่รู้จัก engine {name!r} เลือกจาก {list(ENGINES)}")
        return 1
    mode, engine, use_llm = ENGINES[name]
    docs = sorted(p for p in DATA.iterdir() if p.suffix.lower() in {".pdf", ".jpg", ".jpeg", ".png"})
    if not docs:
        print("ยังไม่มีชุดข้อมูล — รัน prepare ก่อน")
        return 1

    pipe = FinanceOCRPipeline(load_config(), mode=mode, engine=engine, use_llm=use_llm)
    pipe.process_file(docs[0])  # อุ่นเครื่อง โหลดโมเดล ไม่นับเวลา

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{name}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for i, doc in enumerate(docs, 1):
            t0 = time.perf_counter()
            try:
                results = pipe.process_file(doc)
                data, status = _merge_pages(results)
                rec = {
                    "doc": doc.stem, "status": status,
                    "data": json.loads(data.model_dump_json()), "error": None,
                }
            except Exception as e:  # noqa: BLE001 — เอกสารใบหนึ่งพังต้องไม่ล้มทั้ง benchmark
                rec = {"doc": doc.stem, "status": "error", "data": None,
                       "error": f"{type(e).__name__}: {e}"[:300]}
            rec["seconds"] = round(time.perf_counter() - t0, 2)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(docs)}] {doc.stem:42} {rec['status']:13} {rec['seconds']:6.1f}s", flush=True)
    print(f"บันทึกที่ {out}")
    return 0


# ---------------------------------------------------------------- report


def _norm_text(s) -> str:
    return re.sub(r"[^0-9a-z฀-๿]", "", str(s or "").lower()).replace("ํา", "ำ")


def _pred_value(data: dict, field: str):
    if data is None:
        return None
    if field == "vendor_name":
        return (data.get("vendor") or {}).get("name")
    if field == "invoice_number":
        return data.get("invoice_no")
    if field == "invoice_date":
        return data.get("date")
    if field == "tax":
        return data.get("vat_amount")
    if field == "total":
        return data.get("total_amount")
    if field == "subtotal":
        # "subtotal" ของชุดข้อมูล = ยอดก่อนภาษี ซึ่งในสคีมาเราอยู่ได้สองช่อง
        return [data.get("amount_before_vat"), data.get("subtotal")]
    return None


def _field_ok(field: str, want, got) -> bool:
    if field in MONEY:
        cands = got if isinstance(got, list) else [got]
        cands = [None if c is None else float(c) for c in cands]  # ยอดเงินใน JSON เป็นสตริง Decimal ("47.56")
        if want in (None, 0, 0.0):
            return all(c in (None, 0, 0.0) for c in cands) or 0 in cands
        return any(c is not None and abs(float(c) - float(want)) <= 0.011 for c in cands)
    if want in (None, ""):
        return got in (None, "")
    if got in (None, ""):
        return False
    if field == "vendor_name":
        a, b = _norm_text(want), _norm_text(got)
        return a == b or difflib.SequenceMatcher(None, a, b).ratio() >= 0.9
    return _norm_text(want) == _norm_text(got)


def _items_f1(want: list, got: list) -> float | None:
    w = collections.Counter(round(float(x["amount"]), 2) for x in want or [] if x.get("amount") is not None)
    g = collections.Counter(round(float(x["amount"]), 2) for x in got or [] if x.get("amount") is not None)
    if not w and not g:
        return None
    hit = sum((w & g).values())
    if not hit:
        return 0.0
    p, r = hit / sum(g.values()), hit / sum(w.values())
    return 2 * p * r / (p + r)


def report() -> int:
    runs = {p.stem: [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
            for p in sorted(OUT.glob("*.jsonl"))}
    if not runs:
        print("ยังไม่มีผล — รัน run ก่อน")
        return 1
    truth = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in DATA.glob("*.json")}

    lines = ["# ผล benchmark", ""]
    n_docs = len(truth)
    cats = collections.Counter(t["category"] for t in truth.values())
    lines += [f"เอกสาร {n_docs} ฉบับ: " + ", ".join(f"{c} {n}" for c, n in sorted(cats.items())), ""]

    summary, by_cat, by_field = {}, {}, {}
    for name, recs in runs.items():
        field_hits, f1s, times = [], [], []
        per_field = collections.defaultdict(list)
        per_cat = collections.defaultdict(list)
        stp = escaped = errors = 0
        for rec in recs:
            t = truth.get(rec["doc"])
            if t is None:
                continue
            exp = t["expected"]
            times.append(rec["seconds"])
            if rec["status"] == "error":
                errors += 1
            doc_hits = []
            for f in FIELDS:
                if f not in exp.get("scored", FIELDS):
                    continue
                ok = _field_ok(f, exp.get(f), _pred_value(rec["data"], f))
                doc_hits.append(ok)
                per_field[f].append(ok)
            field_hits += doc_hits
            per_cat[t["category"]].append(sum(doc_hits) / len(doc_hits) if doc_hits else 0)
            if rec["data"] is not None:
                f1 = _items_f1(exp.get("lines"), rec["data"].get("items"))
                if f1 is not None:
                    f1s.append(f1)
            auto = rec["status"] == "auto_approved"
            all_ok = all(doc_hits)
            stp += auto
            # ผ่านอัตโนมัติทั้งที่มีช่องผิด หรือทั้งที่เฉลยบอกว่าควรให้คนดู
            escaped += auto and (not all_ok or bool(exp.get("should_flag")))
        n = len(times)
        summary[name] = {
            "acc": sum(field_hits) / len(field_hits) if field_hits else 0,
            "f1": statistics.mean(f1s) if f1s else 0,
            "stp": stp / n, "escaped": escaped / n, "errors": errors,
            "p50": statistics.median(times), "n": n,
        }
        by_cat[name] = {c: statistics.mean(v) for c, v in per_cat.items()}
        by_field[name] = {f: sum(v) / len(v) for f, v in per_field.items()}

    order = sorted(summary, key=lambda k: -summary[k]["acc"])
    lines += [
        "## ภาพรวม",
        "",
        "| engine | ฟิลด์ถูก | F1 รายการ | ผ่านอัตโนมัติ | ผ่านอัตโนมัติทั้งที่ผิด | error | เวลา p50 |",
        "|---|---|---|---|---|---|---|",
    ]
    for k in order:
        s = summary[k]
        lines.append(
            f"| **{k}** | {s['acc']:.1%} | {s['f1']:.2f} | {s['stp']:.0%} | "
            f"{s['escaped']:.0%} | {s['errors']} | {s['p50']:.1f}s |"
        )
    cat_names = sorted({c for v in by_cat.values() for c in v})
    lines += ["", "## ฟิลด์ถูกแยกตามแพทเทิร์น", "",
              "| แพทเทิร์น | " + " | ".join(order) + " |",
              "|---|" + "---|" * len(order)]
    for c in cat_names:
        cells = []
        best = max(by_cat[k].get(c, 0) for k in order)
        for k in order:
            v = by_cat[k].get(c)
            txt = "—" if v is None else f"{v:.0%}"
            cells.append(f"**{txt}**" if v is not None and v == best else txt)
        lines.append(f"| {c} ({cats[c]}) | " + " | ".join(cells) + " |")
    lines += ["", "## ฟิลด์ถูกแยกตามช่อง", "",
              "| ช่อง | " + " | ".join(order) + " |",
              "|---|" + "---|" * len(order)]
    for f in FIELDS:
        lines.append(f"| {f} | " + " | ".join(f"{by_field[k].get(f, 0):.0%}" for k in order) + " |")

    text = "\n".join(lines) + "\n"
    (OUT / "report.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"บันทึกที่ {OUT / 'report.md'}")
    return 0


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    logging.disable(logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="โหลดชุดข้อมูล")
    p.add_argument("--per-cell", type=int, default=2, help="กี่ใบต่อคู่หมวด×ความยาก")
    p.add_argument("--thai", type=Path, help="โฟลเดอร์เอกสารไทยที่มีเฉลยแบบเก่า (.json คู่กับเอกสาร)")
    r = sub.add_parser("run", help="รัน engine หนึ่งตัวกับทุกเอกสาร")
    r.add_argument("engine", choices=list(ENGINES))
    sub.add_parser("report", help="สรุปผลทุก engine")
    a = ap.parse_args()
    if a.cmd == "prepare":
        return prepare(a.per_cell, a.thai)
    if a.cmd == "run":
        return run(a.engine)
    return report()


if __name__ == "__main__":
    raise SystemExit(main())
