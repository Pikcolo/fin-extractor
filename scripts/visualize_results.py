"""ทำรายงาน HTML ให้ดูผลของ engine หนึ่งตัวต่อเอกสาร: ภาพหลัง preprocess + ข้อความดิบ + ค่าที่สกัดพร้อมระดับหลักฐานรายฟิลด์

อ่านเฉพาะผลที่เก็บไว้แล้ว (ไม่เรียก OCR/VLM/LLM ใช้ CPU เบา ๆ) ถ้ามีหลาย extractor (rules, llm) แสดงเรียงต่อกันในหน้าเดียวกัน

    .venv/Scripts/python scripts/visualize_results.py [paddle typhoon ...] [--dataset finance_doc_scan] [--profile default]

ผล: result_ocr/<engine>/<dataset>/visualize/index.html (+ <ไฟล์>/index.html และ page.jpg ต่อเอกสาร) เปิดด้วยเบราว์เซอร์
กรอบ bbox ของ Paddle ไม่อยู่ในผลที่เก็บ (เก็บแค่ข้อความ) ต้องรัน scripts/visualize_preprocess.py ซึ่งอ่านภาพใหม่ด้วย Paddle
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fin_extractor.paths import DEFAULT_PROFILE, DataPaths, find_dir, iter_results  # noqa: E402

MAX_SIDE = 1400
RAW_EXT = {"paddle": "txt", "typhoon": "md", "oxr": "md"}
LEVELS = {  # ลำดับ = ลำดับแสดง (ต้องดูก่อน → ท้ายสุด) ป้ายมีทั้งสัญลักษณ์และข้อความ ไม่พึ่งสีอย่างเดียว
    "conflict": ("✗", "ขัดแย้ง ต้องแก้หรือยืนยัน"),
    "missing": ("?", "หาไม่เจอ"),
    "unverified": ("~", "ไม่มีตัวยืนยัน ดูเทียบภาพ"),
    "verified": ("✓", "มี check ยืนยัน"),
    "not_applicable": ("–", "ไม่มีในเอกสารชนิดนี้"),
}
STATUS_TH = {"auto_approved": "อนุมัติอัตโนมัติ", "needs_review": "ต้องตรวจ", "failed": "ล้มเหลว"}

CSS = """
:root{--bg:#fff;--fg:#1d2433;--muted:#5b6475;--card:#f5f6f8;--line:#d9dce3;
--conflict:#d55e00;--missing:#cc79a7;--unverified:#8a6100;--verified:#00785a;--not_applicable:#5b6475;
--auto_approved:#00785a;--needs_review:#8a6100;--failed:#d55e00}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#14171f;--fg:#e6e8ee;--muted:#a0a7b7;--card:#1e222c;--line:#343a48;
--conflict:#ff9a57;--missing:#e6a0c4;--unverified:#e6b84d;--verified:#4fd1a5;--not_applicable:#a0a7b7;
--auto_approved:#4fd1a5;--needs_review:#e6b84d;--failed:#ff9a57}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,"Segoe UI",sans-serif}
main{max-width:1200px;margin:0 auto;padding:16px}
h1{font-size:1.4rem;margin:.4em 0} h2{font-size:1.1rem;margin:1.6rem 0 .4rem;border-top:1px solid var(--line);padding-top:1rem}
.muted{color:var(--muted);font-size:.9rem} a{color:inherit}
.cols{display:grid;grid-template-columns:minmax(0,5fr) minmax(0,7fr);gap:16px;align-items:start}
@media (max-width:820px){.cols{grid-template-columns:1fr}}
img{max-width:100%;height:auto;border:1px solid var(--line)}
.tw{overflow-x:auto} table{border-collapse:collapse;width:100%;font-size:.92rem}
th,td{border:1px solid var(--line);padding:4px 8px;text-align:left;vertical-align:top} th{background:var(--card)}
.chip{display:inline-block;border:1.5px solid currentColor;border-radius:999px;padding:0 9px;font-size:.82rem;white-space:nowrap}
pre{white-space:pre-wrap;word-break:break-word;background:var(--card);padding:8px;font:.85rem/1.5 ui-monospace,Consolas,monospace;max-height:520px;overflow:auto}
.sum span{display:inline-block;margin-right:16px}
"""


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def chip(kind: str, symbol: str, text: str) -> str:
    return f'<span class="chip" style="color:var(--{kind})">{symbol} {esc(text)}</span>'


def page(title: str, body: str) -> str:
    return (f'<!doctype html><html lang="th"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{esc(title)}</title><style>{CSS}</style></head><body><main>{body}</main></body></html>")


def load_results(final_dir: Path) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in iter_results(final_dir, "*_p*.json")}


def doc_name(stem: str) -> str:
    return re.sub(r"_p\d+$", "", stem)


def fields_table(r: dict) -> str:
    order = list(LEVELS)
    rows = sorted(r["field_evidence"].items(), key=lambda kv: (order.index(kv[1]["level"]), kv[0]))
    out = []
    for key, ev in rows:
        sym, label = LEVELS[ev["level"]]
        notes = [*(f"✓ {c}" for c in ev["checks_passed"]), *(f"✗ {c}" for c in ev["checks_failed"])]
        if ev.get("sources_agree") is False:
            notes.append("rules ≠ llm")
        if ev.get("grounded") is False:
            notes.append("ไม่พบในข้อความ")
        out.append(f"<tr><td><code>{esc(key)}</code></td><td>{esc(ev['value'])}</td><td>{chip(ev['level'], sym, label)}</td>"
                   f"<td>{esc(ev.get('source') or '')}</td><td>{esc(', '.join(notes))}</td></tr>")
    return ('<div class="tw"><table><tr><th>ฟิลด์</th><th>ค่า</th><th>ระดับหลักฐาน</th><th>ตัวสกัด</th><th>ตรวจ</th></tr>'
            + "".join(out) + "</table></div>")


def checks_table(r: dict) -> str:
    outcome = {True: ("verified", "✓", "ผ่าน"), False: ("failed", "✗", "ไม่ผ่าน"), None: ("not_applicable", "·", "ไม่ได้ตรวจ")}
    rows = "".join(f"<tr><td>{esc(c['name'])}</td><td>{chip(*outcome[c['passed']])}</td><td>{esc(c['detail'])}</td></tr>"
                   for c in r["checks"])
    return f'<div class="tw"><table><tr><th>check</th><th>ผล</th><th>รายละเอียด</th></tr>{rows}</table></div>'


def save_page_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((MAX_SIDE, MAX_SIDE))
        im.save(dst, quality=85)


def build(engine: str, dataset: str, profile: str) -> Path:
    P = DataPaths()
    base, raw_dir = P.engine(engine, dataset), P.raw_data(engine, dataset)
    extractors = sorted(d.name for d in (base / "final").glob("*") if d.is_dir())
    if not extractors:
        raise SystemExit(f"ไม่มีผลสกัดของ {engine}/{dataset} ที่ {base / 'final'}")
    results = {x: load_results(P.final(engine, dataset, x)) for x in extractors}
    stems = sorted({s for r in results.values() for s in r})
    out = P.visualize(engine, dataset)
    pre_final = P.preprocess_final(dataset, profile)
    ext = RAW_EXT.get(engine, "txt")

    rows = []
    for stem in stems:
        name = doc_name(stem)
        src = find_dir(pre_final, name) / f"{stem}.png"
        if src.exists():
            save_page_image(src, out / name / "page.jpg")
        raw_path = find_dir(raw_dir, name) / f"{stem}.{ext}"
        raw = raw_path.read_text(encoding="utf-8") if raw_path.exists() else "(ไม่พบข้อความดิบ)"
        sections = []
        for x in extractors:
            r = results[x].get(stem)
            if not r:
                continue
            kind = r["status"]
            sections.append(f'<h2>{esc(x)} {chip(kind, "●", STATUS_TH.get(kind, kind))}</h2>'
                            f'<p class="muted">ขาด: {esc(", ".join(r["missing_fields"]) or "-")} · engine: {esc(r["engine"])}</p>'
                            + fields_table(r) + "<h3>checks</h3>" + checks_table(r))
        img = '<img src="page.jpg" alt="ภาพหลัง preprocess">' if src.exists() else '<p class="muted">ไม่พบภาพ preprocess</p>'
        body = (f'<p><a href="../index.html">← ทุกเอกสาร</a></p><h1>{esc(name)} <span class="muted">{esc(engine)}</span></h1>'
                f'<div class="cols"><div>{img}</div><div>{"".join(sections)}'
                f'<h2>ข้อความดิบ ({esc(engine)})</h2><pre>{esc(raw)}</pre></div></div>')
        (out / name).mkdir(parents=True, exist_ok=True)
        (out / name / "index.html").write_text(page(f"{name} · {engine}", body), encoding="utf-8")
        cells = "".join(f"<td>{chip(results[x][stem]['status'], '●', STATUS_TH.get(results[x][stem]['status'], ''))}</td>"
                        if stem in results[x] else "<td>-</td>" for x in extractors)
        first = next(results[x][stem] for x in extractors if stem in results[x])
        rows.append(f'<tr><td><a href="{esc(name)}/index.html">{esc(name)}</a></td><td>{esc(first["data"].get("document_type"))}</td>{cells}'
                    f'<td>{esc(", ".join(first["missing_fields"]) or "-")}</td></tr>')

    summary = "".join(f'<p class="sum"><b>{esc(x)}</b> ' + "".join(
        f"<span>{chip(k, '●', STATUS_TH.get(k, k))} {n}</span>" for k, n in Counter(r["status"] for r in results[x].values()).items()) + "</p>"
        for x in extractors)
    head = "".join(f"<th>{esc(x)}</th>" for x in extractors)
    body = (f"<h1>{esc(engine)} · {esc(dataset)}</h1><p class='muted'>{len(stems)} เอกสาร · ภาพ = หลัง preprocess (profile {esc(profile)}) · "
            f"ระดับหลักฐานรายฟิลด์ในหน้าแต่ละเอกสาร (✗ ขัดแย้ง ? หาไม่เจอ ~ ไม่มีตัวยืนยัน ✓ มี check ยืนยัน)</p>{summary}"
            f'<div class="tw"><table><tr><th>เอกสาร</th><th>ชนิด</th>{head}<th>ขาด (extractor แรก)</th></tr>{"".join(rows)}</table></div>')
    (out / "index.html").write_text(page(f"{engine} · {dataset}", body), encoding="utf-8")
    return out / "index.html"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("engines", nargs="*", default=["paddle", "typhoon"])
    ap.add_argument("--dataset", default="finance_doc_scan")
    ap.add_argument("--profile", default=DEFAULT_PROFILE)
    args = ap.parse_args()
    for e in args.engines:
        print(build(e, args.dataset, args.profile))


if __name__ == "__main__":
    main()
