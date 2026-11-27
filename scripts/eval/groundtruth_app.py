"""เครื่องมือกรอก ground truth ในเครื่อง: เปิดภาพเอกสารคู่กับฟอร์มฟิลด์ เติมค่าจากผลสกัดไว้ก่อน คนยืนยันหรือแก้ แล้วบันทึก

    .venv/Scripts/python scripts/eval/groundtruth_app.py <โฟลเดอร์อินพุต>
    .venv/Scripts/python scripts/eval/groundtruth_app.py <โฟลเดอร์อินพุต> --prefill typhoon/luma/rules --port 8765

เปิด http://127.0.0.1:8765 ในเบราว์เซอร์ (ผูกกับ 127.0.0.1 เท่านั้น เครื่องอื่นในเครือข่ายเข้าไม่ได้ ไม่มีอะไรส่งออกนอกเครื่อง)
- ภาพ = ภาพ final ของขั้น 2 (``02_preprocess/<dataset>/<profile>/final/``) ฟอร์ม = ผลสกัดของ ``--prefill`` (ค่าตั้งต้น openthai/regions-luma/rules = flow หลัก)
- ช่องว่าง + ติ๊ก "ไม่มี" = เอกสารไม่มีช่องนี้ (null) · บันทึก = ยืนยันทุกช่องในฟอร์ม
- "บันทึกร่าง" (draft) ยังไม่นับคะแนน "ตรวจเสร็จ" (done) นับ ดูคะแนนด้วย scripts/eval/score_groundtruth.py
ผลที่ ``data/etc/groundtruth/<dataset>/<ไฟล์>.json`` (รูปแบบดู src/fin_extractor/groundtruth.py)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fin_extractor import groundtruth as G  # noqa: E402
from fin_extractor.config import load_config  # noqa: E402
from fin_extractor.doctypes import load_profiles  # noqa: E402
from fin_extractor.paths import DEFAULT_PROFILE, DataPaths, dataset_of  # noqa: E402
from fin_extractor.sections.build import page_images  # noqa: E402

LABELS = {
    "vendor.name": "ผู้ขาย/ผู้ออกเอกสาร: ชื่อ", "vendor.tax_id": "ผู้ขาย: เลขผู้เสียภาษี", "vendor.branch": "ผู้ขาย: สาขา", "vendor.address": "ผู้ขาย: ที่อยู่",
    "buyer.name": "ผู้ซื้อ/ลูกค้า: ชื่อ", "buyer.tax_id": "ผู้ซื้อ: เลขผู้เสียภาษี", "buyer.branch": "ผู้ซื้อ: สาขา", "buyer.address": "ผู้ซื้อ: ที่อยู่",
    "book_no": "เล่มที่", "invoice_no": "เลขที่เอกสาร", "date": "วันที่ (YYYY-MM-DD ค.ศ.)",
    "subtotal": "รวมเงินก่อนส่วนลด", "discount": "ส่วนลด", "amount_before_vat": "มูลค่าก่อน VAT (เฉพาะส่วนที่เสียภาษี)", "non_taxable_amount": "มูลค่าที่ไม่เสียภาษี (ยกเว้นภาษี)", "vat_rate": "อัตรา VAT (เช่น 0.07)",
    "vat_amount": "VAT", "total_amount": "ยอดสุทธิรวม VAT",
}


class App:
    def __init__(self, input_path: str, prefill_run: str, profile: str = DEFAULT_PROFILE):
        self.cfg = load_config()
        self.paths = DataPaths(self.cfg)
        self.dataset = dataset_of(input_path)
        self.run = prefill_run
        self.pages = page_images(self.paths.preprocess_final(self.dataset, profile))
        if not self.pages:
            sys.exit(f"ไม่พบภาพ final ใน {self.paths.preprocess_final(self.dataset, profile)} — รัน sections <input> --to preprocess ก่อน")
        self.profiles = {p.name: p for p in load_profiles(self.cfg.get("document_types") or {})}

    def docs(self) -> list[dict]:
        out = []
        for stem in self.pages:
            gt = G.load_gt(self.paths, self.dataset, stem)
            out.append({"stem": stem, "status": gt.get("status", "draft") if gt else "new"})
        return out

    def doc(self, stem: str) -> dict:
        result = G.load_result(self.paths, self.dataset, self.run, stem)
        gt = G.load_gt(self.paths, self.dataset, stem) or G.prefill(result, self.run)
        profile = self.profiles.get(((result or {}).get("data") or {}).get("document_profile") or "")
        specs = [(f.key, f.label) for f in profile.fields] if profile else []
        keys = [k for k, _ in specs] + [k for k in gt.get("document_fields", {}) if k not in dict(specs)]
        system = G.result_values(result)[0] if result else {}
        labels = {**LABELS, **(profile.party_labels() if profile else {})}  # ป้ายฝ่ายตามชื่อในฟอร์มของชนิดนี้ (ถ้ามี)
        return {"stem": stem, "gt": gt, "fields": [{"key": k, "label": labels.get(k, k)} for k in G.FIELDS],
                "document_fields": [{"key": k, "label": dict(specs).get(k, k)} for k in keys],
                "profile": profile.name if profile else None, "prefill_run": self.run if result else None, "system": system}

    def save(self, stem: str, body: dict) -> dict:
        def clean(d: dict, allowed) -> dict:
            return {k: (None if v is None else str(v).strip()) for k, v in (d or {}).items() if allowed is None or k in allowed}

        doc = {"source": stem, "dataset": self.dataset, "page": 1,
               "status": "done" if body.get("status") == "done" else "draft",
               "prefill_from": body.get("prefill_from"), "updated": datetime.now().isoformat(timespec="seconds"),
               "note": str(body.get("note") or ""),
               "fields": clean(body.get("fields"), set(G.FIELDS)),
               "document_fields": {k: v for k, v in clean(body.get("document_fields"), None).items() if k.strip()},
               "items": [clean(it, set(G.ITEM_KEYS)) for it in body.get("items") or [] if any(str(v or "").strip() for v in it.values())],
               "items_checked": bool(body.get("items_checked"))}
        G.save_gt(self.paths, self.dataset, stem, doc)
        return {"ok": True, "status": doc["status"]}


def handler(app: App):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # ไม่พิมพ์ทุกคำขอ (มีชื่อไฟล์เอกสาร)
            pass

        def send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def json(self, obj, code=200):
            self.send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def stem(self, path: str, prefix: str) -> str | None:
            s = unquote(path[len(prefix):])
            return s if s in app.pages else None  # รับเฉพาะชื่อไฟล์ที่มีอยู่ (กันพาธแปลก ๆ)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                return self.send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            if path == "/api/docs":
                return self.json({"dataset": app.dataset, "prefill": app.run, "docs": app.docs()})
            if path.startswith("/api/doc/") and (s := self.stem(path, "/api/doc/")):
                return self.json(app.doc(s))
            if path.startswith("/img/") and (s := self.stem(path, "/img/")):
                return self.send(200, app.pages[s].read_bytes(), "image/png")
            self.json({"error": "not found"}, 404)

        def do_POST(self):
            path = urlparse(self.path).path
            if path.startswith("/api/doc/") and (s := self.stem(path, "/api/doc/")):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                return self.json(app.save(s, body))
            self.json({"error": "not found"}, 404)

    return H


PAGE = r"""<!doctype html>
<html lang="th"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ground truth</title>
<style>
:root { --bg:#f6f6f4; --panel:#fff; --ink:#1d1d1b; --muted:#6b6b66; --line:#dcdcd6; --accent:#2f5d8a; --ok:#2e7d4f; --warn:#a15c00; --diff:#fff4d6; }
* { box-sizing:border-box; }
body { margin:0; font:14px/1.45 system-ui, "Segoe UI", Tahoma, sans-serif; color:var(--ink); background:var(--bg); height:100vh; display:grid; grid-template-columns:190px 1fr 470px; }
#list { overflow:auto; border-right:1px solid var(--line); background:var(--panel); }
#list h1 { font-size:13px; margin:10px 12px; color:var(--muted); font-weight:600; }
#list button { display:flex; justify-content:space-between; width:100%; border:0; background:none; text-align:left; padding:7px 12px; cursor:pointer; font:inherit; }
#list button.cur { background:#e8eef5; }
.badge { font-size:11px; padding:1px 6px; border-radius:8px; background:#eee; color:var(--muted); }
.badge.done { background:#dff0e5; color:var(--ok); } .badge.draft { background:#fdf0dc; color:var(--warn); }
#view { overflow:auto; background:#e9e9e5; position:relative; }
#bar { position:sticky; top:0; display:flex; gap:8px; align-items:center; padding:6px 10px; background:rgba(246,246,244,.95); border-bottom:1px solid var(--line); z-index:1; }
#img { display:block; margin:10px auto; box-shadow:0 1px 4px rgba(0,0,0,.2); background:#fff; }
#form { overflow:auto; background:var(--panel); border-left:1px solid var(--line); padding:10px 14px 80px; }
#form h2 { font-size:15px; margin:4px 0 2px; } #form h3 { font-size:13px; margin:16px 0 6px; color:var(--muted); }
.row { display:grid; grid-template-columns:150px 1fr 44px; gap:6px; align-items:center; margin:3px 0; }
.row label { font-size:12px; color:var(--muted); }
.row.changed input[type=text] { background:var(--diff); }
input[type=text], textarea { width:100%; font:inherit; padding:4px 6px; border:1px solid var(--line); border-radius:4px; }
input[type=text]:disabled { background:#f0f0ec; color:#aaa; }
.na { font-size:11px; color:var(--muted); display:flex; gap:2px; align-items:center; }
.sys { grid-column:2 / 4; font-size:11px; color:var(--muted); margin-top:-2px; }
table { width:100%; border-collapse:collapse; } td { padding:2px; } td input { width:100%; }
.actions { position:fixed; bottom:0; right:0; width:470px; display:flex; gap:8px; padding:10px 14px; background:var(--panel); border-top:1px solid var(--line); }
button.primary { background:var(--accent); color:#fff; border:0; } button { padding:6px 10px; border:1px solid var(--line); border-radius:4px; background:#fff; cursor:pointer; font:inherit; }
#msg { font-size:12px; color:var(--muted); align-self:center; }
.help { font-size:12px; color:var(--muted); margin:4px 0 8px; }
</style></head><body>
<nav id="list"><h1 id="ds"></h1><div id="docs"></div></nav>
<main id="view"><div id="bar">ซูม <input id="zoom" type="range" min="30" max="200" value="60"> <span id="zv">60%</span></div><img id="img" alt=""></main>
<section id="form"></section>
<div class="actions"><button onclick="save('draft')">บันทึกร่าง</button><button class="primary" onclick="save('done')">ตรวจเสร็จ บันทึก</button>
<button onclick="go(1)">ถัดไป ›</button><span id="msg"></span></div>
<script>
let docs = [], cur = null, data = null, dirty = false;
const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
async function init() {
  const r = await (await fetch("/api/docs")).json(); docs = r.docs;
  $("#ds").textContent = r.dataset + " · เติมจาก " + r.prefill; renderList(); open(docs[0].stem);
}
function renderList() {
  $("#docs").innerHTML = docs.map(d => `<button class="${d.stem===cur?'cur':''}" onclick="open('${d.stem}')"><span>${esc(d.stem.slice(-4))}</span><span class="badge ${d.status}">${{new:'ยังไม่ทำ',draft:'ร่าง',done:'เสร็จ'}[d.status]}</span></button>`).join("");
}
async function open(stem) {
  if (dirty && !confirm("ยังไม่ได้บันทึก ทิ้งการแก้ไข?")) return;
  cur = stem; data = await (await fetch("/api/doc/" + encodeURIComponent(stem))).json(); dirty = false;
  $("#img").src = "/img/" + encodeURIComponent(stem); zoom(); renderList(); renderForm(); $("#msg").textContent = "";
}
function fieldRow(group, f, val, sys) {
  const na = val === null;
  const shown = sys !== undefined && (sys ?? "") !== (val ?? "") ? `<div class="sys">ระบบอ่านได้: ${esc(sys ?? "—")}</div>` : "";
  return `<div class="row${shown ? ' changed' : ''}" data-group="${group}" data-key="${esc(f.key)}"><label title="${esc(f.key)}">${esc(f.label)}</label>
    <input type="text" value="${esc(val)}" ${na ? "disabled" : ""} oninput="dirty=true">
    <span class="na"><input type="checkbox" ${na ? "checked" : ""} onchange="toggleNa(this)">ไม่มี</span>${shown}</div>`;
}
function renderForm() {
  const g = data.gt, sys = data.system || {};
  let h = `<h2>${esc(data.stem)}</h2><div class="help">โปรไฟล์: ${esc(data.profile || "—")} · เติมจาก ${esc(data.prefill_run || "— (ไม่มีผลสกัด)")} · สถานะ ${esc(g.status)}<br>
    แก้ค่าให้ตรงกับเอกสาร · ช่องที่เอกสารไม่มี ติ๊ก "ไม่มี" · ช่องสีเหลือง = ต่างจากที่ระบบอ่านได้</div><h3>ฟิลด์หลัก</h3>`;
  h += data.fields.map(f => fieldRow("fields", f, g.fields[f.key] === undefined ? null : g.fields[f.key], sys[f.key])).join("");
  h += `<h3>ฟิลด์ประจำชนิดเอกสาร</h3>` + (data.document_fields.length ? data.document_fields.map(f => fieldRow("document_fields", f, (g.document_fields||{})[f.key] ?? null, undefined)).join("") : `<div class="help">ไม่มี</div>`);
  h += `<h3>รายการสินค้า/บริการ</h3><label class="na"><input type="checkbox" id="items_checked" ${g.items_checked ? "checked" : ""} onchange="dirty=true">ตรวจรายการแล้ว (นับคะแนนรายการ)</label>
    <table id="items"><tr><td>รายละเอียด</td><td style="width:60px">จำนวน</td><td style="width:80px">ราคา/หน่วย</td><td style="width:80px">จำนวนเงิน</td><td></td></tr>
    ${(g.items||[]).map(itemRow).join("")}</table><button onclick="addItem()">+ เพิ่มรายการ</button>
    <h3>หมายเหตุ</h3><textarea id="note" rows="3" oninput="dirty=true">${esc(g.note)}</textarea>`;
  $("#form").innerHTML = h;
}
function itemRow(it) {
  it = it || {};
  return `<tr>${["description","quantity","unit_price","amount"].map(k => `<td><input type="text" data-k="${k}" value="${esc(it[k])}" oninput="dirty=true"></td>`).join("")}<td><button onclick="this.closest('tr').remove();dirty=true">×</button></td></tr>`;
}
function addItem() { $("#items").insertAdjacentHTML("beforeend", itemRow()); dirty = true; }
function toggleNa(cb) { const inp = cb.closest(".row").querySelector("input[type=text]"); inp.disabled = cb.checked; if (cb.checked) inp.value = ""; dirty = true; }
function collect(group) {
  const out = {};
  document.querySelectorAll(`.row[data-group="${group}"]`).forEach(r => {
    const inp = r.querySelector("input[type=text]"), na = r.querySelector("input[type=checkbox]").checked;
    out[r.dataset.key] = na || inp.value.trim() === "" ? null : inp.value.trim();
  });
  return out;
}
async function save(status) {
  const items = [...document.querySelectorAll("#items tr")].slice(1).map(tr => Object.fromEntries([...tr.querySelectorAll("input")].map(i => [i.dataset.k, i.value.trim() || null])));
  const body = {status, prefill_from: data.gt.prefill_from, fields: collect("fields"), document_fields: collect("document_fields"),
    items, items_checked: $("#items_checked").checked, note: $("#note").value};
  const r = await (await fetch("/api/doc/" + encodeURIComponent(cur), {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)})).json();
  dirty = false; docs.find(d => d.stem === cur).status = r.status; renderList();
  $("#msg").textContent = "บันทึกแล้ว (" + (r.status === "done" ? "เสร็จ" : "ร่าง") + ") " + new Date().toLocaleTimeString();
}
function go(step) { const i = docs.findIndex(d => d.stem === cur); if (docs[i + step]) open(docs[i + step].stem); }
function zoom() { const z = $("#zoom").value; $("#zv").textContent = z + "%"; const im = $("#img"); im.style.width = im.naturalWidth ? (im.naturalWidth * z / 100) + "px" : z + "%"; }
$("#zoom").oninput = zoom; $("#img").onload = zoom;
window.onbeforeunload = e => dirty ? e.preventDefault() : undefined;
init();
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="โฟลเดอร์อินพุต (ใช้เอาชื่อชุดข้อมูล)")
    ap.add_argument("--prefill", default="openthai/regions-luma/rules", help="ผลสกัดที่ใช้เติมฟอร์ม <engine>/<วิธี>/<extractor>")
    ap.add_argument("--profile", default=DEFAULT_PROFILE)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    app = App(args.input, args.prefill, args.profile)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(app))
    print(f"เปิด http://127.0.0.1:{args.port}  ({len(app.pages)} ไฟล์ · บันทึกที่ {G.gt_dir(app.paths, app.dataset)}) Ctrl+C เพื่อปิด", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
