"""หน้าเว็บลอง OCR ทีละขั้น: อัปโหลดเอกสาร 1 ไฟล์ แล้วดูผลของทุกขั้นใน flow หลักจนได้ JSON สุดท้าย

    .venv/Scripts/python -X utf8 scripts/demo/ocr_demo_app.py
    .venv/Scripts/python -X utf8 scripts/demo/ocr_demo_app.py --port 8766

เปิด http://127.0.0.1:8766 (ผูกกับ 127.0.0.1 เท่านั้น ไม่มีอะไรส่งออกนอกเครื่อง ยกเว้นเลือก engine typhoon ที่ตั้ง provider เป็น API)
ขั้นที่แสดง = flow ของคำสั่ง ``sections`` ด้วยค่าตั้งต้นเสมอ (``FLOW_DEFAULTS``: luma, table_mode rows, ระบาย QR) เลือกได้แค่ตัวอ่าน ใช้ฟังก์ชันเดียวกับ pipeline จริง ไม่ได้เขียนตรรกะใหม่
    1 อินพุต → 2 preprocess* → 3 ผังแบ่งส่วน → 4 ภาพรายส่วน → 5 OCR ทีละส่วน* → 6 รวมข้อความ → 7 สกัดฟิลด์ → 8 ตรวจ/สถานะ → 9 JSON
    (* รันใน ``ocr_demo_worker.py`` แยก process: Paddle กับ torch อยู่ process เดียวกันไม่ได้ และโมเดลล้มแล้วเซิร์ฟเวอร์ไม่ล้มตาม)
ขั้น 3 วาดภาพกลางทาง (เส้นตาราง หมึก จุดตัด) จากฟังก์ชันใน ``sections/layout.py`` ส่วนผังจริงมาจาก ``layout_sections`` ตัวเดียวกับ flow หลัก
อ่านเฉพาะหน้าแรกของ PDF · รันได้ทีละงาน (กัน GPU สองงานพร้อมกัน) · ผลของแต่ละงานอยู่ที่ ``data/etc/demo/<งาน>/``
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import subprocess
import sys
import threading
import time
import traceback
import uuid
from datetime import date
from decimal import Decimal, InvalidOperation
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from fin_extractor.config import load_config  # noqa: E402
from fin_extractor.doctypes import DOC_PREFIX, squash  # noqa: E402
from fin_extractor.extraction import lines_with_chunks  # noqa: E402
from fin_extractor.extraction.patterns import find_dates, find_numbers, find_percents  # noqa: E402
from fin_extractor.io import load_pages, read_image, save_image  # noqa: E402
from fin_extractor.io.loader import SUPPORTED_EXTS  # noqa: E402
from fin_extractor.paths import DataPaths  # noqa: E402
from fin_extractor.pipeline import FinanceOCRPipeline  # noqa: E402
from fin_extractor.preprocessing import binarize as bin_ops  # noqa: E402
from fin_extractor.preprocessing import enhance  # noqa: E402
from fin_extractor.readers import ENGINES  # noqa: E402
from fin_extractor.sections import graphics as G  # noqa: E402
from fin_extractor.sections import images as I  # noqa: E402,E741
from fin_extractor.sections import layout as L  # noqa: E402
from fin_extractor.sections.build import write_png  # noqa: E402
from fin_extractor.sections.read import markdown_of  # noqa: E402
from fin_extractor.validation.evidence import flatten  # noqa: E402

WORKER = Path(__file__).with_name("ocr_demo_worker.py")
PAGE = Path(__file__).with_name("ocr_demo.html")
FONTS = ROOT / "assets" / "fonts"
MAX_UPLOAD = 50 * 2**20
# ตั้งค่าของขั้น preprocess/แบ่งส่วน = ค่าตั้งต้นของคำสั่ง sections (ไม่ให้เลือกบนหน้าเว็บ): ตรวจทิศอัตโนมัติ · ไม่ใช้ --fill-aware / --dotted-rules · ระบาย QR/บาร์โค้ด
FLOW_DEFAULTS = {"orientation": "auto", "fill": False, "dotted": False, "codes": True, "trim": None}  # trim = --trim-sections (ไม่หดเป็นค่าตั้งต้น)

STAGES = [("input", "อินพุต"), ("preprocess", "ปรับภาพ (preprocess)"), ("layout", "แบ่งเป็นส่วน (layout)"),
          ("crops", "ภาพรายส่วนก่อน OCR"), ("ocr", "OCR ทีละส่วน"), ("text", "รวมข้อความ → บรรทัด"),
          ("extract", "สกัดฟิลด์ (rules)"), ("validate", "ตรวจความสอดคล้อง + สถานะ"), ("result", "JSON สุดท้าย")]

state_lock = threading.RLock()  # กันอ่านสถานะงาน (หน้าเว็บ poll) ระหว่างที่เธรดงานกำลังแก้ (RLock: ขั้น 7–9 ประกาศพร้อมกันในล็อกเดียว)
run_lock = threading.Lock()  # ทีละงาน: OCR สองงานพร้อมกันบน GPU 6 GB ไม่ไหว
JOBS: dict[str, dict] = {}


class StageError(Exception):
    pass


# ---------------------------------------------------------------- ภาพ

def save_view(path: Path, img: np.ndarray, side: int = 1400) -> None:
    s = min(1.0, side / max(img.shape[:2]))
    save_image(path, cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img)


def ink_view(mask: np.ndarray) -> np.ndarray:
    """หน้ากากหมึก (255 = หมึก) → ดำบนขาวให้ดูง่าย."""
    return 255 - mask


def lines_view(horiz: np.ndarray, vert: np.ndarray, bw: np.ndarray) -> np.ndarray:
    """ภาพขาวดำจาง ๆ + เส้นแนวนอนสีแดง เส้นตั้งสีน้ำเงิน."""
    out = cv2.cvtColor(np.where(bw > 0, 255, 215).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    out[horiz > 0] = (40, 40, 220)
    out[vert > 0] = (220, 90, 20)
    return out


# ---------------------------------------------------------------- ขั้นที่รันใน process นี้ (CPU)

def stage_input(job: dict, d: Path, cfg: dict) -> dict:
    src = next(d.glob("source.*"))  # ไฟล์ที่อัปโหลด (input.jpg ข้าง ๆ เป็นภาพย่อไว้แสดงเท่านั้น)
    pages = load_pages(src, cfg["loader"]["pdf_dpi"])
    save_view(d / "input.jpg", pages[0])
    h, w = pages[0].shape[:2]
    return {"file": job["file"], "ext": src.suffix.lower(), "bytes": src.stat().st_size, "pages": len(pages),
            "size": [w, h], "dpi": cfg["loader"]["pdf_dpi"] if src.suffix.lower() == ".pdf" else None, "image": "input.jpg"}


def stage_layout(job: dict, d: Path, cfg: dict) -> dict:
    """ผังแบ่งส่วนแบบเดียวกับ ``build_layouts`` + ภาพกลางทางของแต่ละขั้นย่อย."""
    opts = job["options"]
    full = read_image(d / "final.png")
    src_dpi = cfg["loader"]["pdf_dpi"] * float(job_stage(job, "preprocess")["data"]["meta"].get("scale", 1.0))
    found = G.find_codes(full, src_dpi) if opts["codes"] else []  # ระบาย QR/บาร์โค้ดก่อนแบ่ง เหมือน build_layouts
    before, scale = L.resize_to_work(full)
    full = G.blank(full, found)
    work, _ = L.resize_to_work(full)
    gray = enhance.to_gray(work)
    otsu_t, _ = cv2.threshold(cv2.GaussianBlur(gray, (3, 3), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = bin_ops.otsu(gray)
    horiz, vert = L.line_masks(bw)
    ink = L.text_ink(bw)
    region, text_on_fill = I.fill_masks(work) if opts["fill"] else (None, None)
    ink_all = cv2.bitwise_or(ink, text_on_fill) if text_on_fill is not None else ink
    cut_ink = cv2.bitwise_or(ink_all, region) if region is not None else ink_all

    raw_tables = L.table_regions(bw)
    tables = L.drop_frames(raw_tables)
    cands: dict[int, str] = {}
    for t in tables:
        tx, ty, tw, th = t
        for y in L.table_rules(horiz, t):
            cands.setdefault(y, "rule")
        cands[ty], cands[ty + th] = "edge", "edge"
    if opts["dotted"]:
        joined = L.thin_joined(bw)
        for t in tables:
            for y in L.cuts_below_rules(cut_ink, joined, L.dotted_rules(joined, t)):
                cands.setdefault(y, "dotted")
    cuts = L.band_cuts(cut_ink, set(cands))
    secs = L.layout_sections(bw, max_piece_frac=L.MAX_PIECE_FRAC, fill_region=region, fill_text=text_on_fill, dotted=opts["dotted"])
    G.mark_sections(secs, found, full.shape, work.shape)
    if opts.get("trim"):
        G.trim_sections(full, work, secs, src_dpi, opts["trim"])

    save_image(d / "lay_work.jpg", work)
    save_image(d / "lay_before.jpg", before)
    save_image(d / "lay_gray.jpg", gray)
    write_png(d / "lay_bw.png", bw)
    write_png(d / "lay_lines.png", lines_view(horiz, vert, bw))
    write_png(d / "lay_ink.png", ink_view(ink_all))
    if region is not None:
        write_png(d / "lay_fill.png", ink_view(region))
    hh, ww = bw.shape
    profile = (cut_ink > 0).mean(axis=1)
    (d / "layout.json").write_text(json.dumps({"layout_sections": secs, "masked_codes": found}, ensure_ascii=False), encoding="utf-8")
    return {
        "work_size": [ww, hh], "full_size": [int(full.shape[1]), int(full.shape[0])], "scale": round(scale, 4), "work_side": L.WORK,
        "otsu": float(otsu_t), "line_kernel": [max(ww // 30, 3), max(hh // 30, 3)],
        "raw_tables": [list(t) for t in raw_tables], "tables": [list(t) for t in tables],
        "candidates": sorted([y, k] for y, k in cands.items()), "cuts": cuts,
        "profile": [round(float(v), 4) for v in profile], "empty_row": L.EMPTY_ROW,
        "max_piece_frac": L.MAX_PIECE_FRAC, "max_piece_px": round(L.MAX_PIECE_FRAC * hh),
        "sections": secs, "fill": region is not None, "dotted": opts["dotted"], "codes_on": opts["codes"], "src_dpi": src_dpi,
        "codes": [{"kind": c["kind"], "data": c["data"], "box": [round(v * scale) for v in c["box"]]} for c in found],
        "images": {"work": "lay_work.jpg", "before": "lay_before.jpg", "gray": "lay_gray.jpg", "bw": "lay_bw.png", "lines": "lay_lines.png",
                   "ink": "lay_ink.png", **({"fill": "lay_fill.png"} if region is not None else {})},
    }


def stage_crops(job: dict, d: Path, cfg: dict) -> dict:
    """ภาพรายส่วนแบบ luma (ค่าตั้งต้นของ flow) ตัดจากภาพ final ความละเอียดเต็มเหมือน ``build_images``."""
    lay = json.loads((d / "layout.json").read_text(encoding="utf-8"))
    full = G.blank(read_image(d / "final.png"), lay["masked_codes"])  # ภาพเดียวกับที่ใช้แบ่ง (QR/บาร์โค้ดระบายแล้ว)
    work, _ = L.resize_to_work(full)
    secs = lay["layout_sections"]
    out = d / "sections"
    out.mkdir(exist_ok=True)
    index, blank = [], []
    for s in secs:
        if not s["keep"]:
            blank.append({"id": s["id"], "kind": s["kind"], "bbox": s["bbox"]})
            continue
        crop = L.crop_full(full, work, s)
        if not crop.size:
            continue
        write_png(out / f"{s['id']}.png", enhance.to_gray(crop))
        save_image(out / f"{s['id']}_color.jpg", crop)
        index.append({"id": s["id"], "kind": s["kind"], "bbox": s["bbox"], "table": s["table"],
                      **({"part": s["part"], "parts": s["parts"]} if s.get("part") else {}),
                      "size": [int(crop.shape[1]), int(crop.shape[0])], "file": f"{s['id']}.png", "color": f"{s['id']}_color.jpg"})
    (out / "index.json").write_text(json.dumps({"sections": index}, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"sections": index, "blank": blank, "variant": "luma"}


def stage_text(job: dict, d: Path, cfg: dict) -> dict:
    secs = json.loads((d / "layout.json").read_text(encoding="utf-8"))["layout_sections"]
    read = {r["id"]: r for r in job_stage(job, "ocr")["data"]["sections"]}
    doc = {"sections": [{"id": s["id"], "kind": s["kind"], **({"part": s["part"], "parts": s["parts"]} if s.get("part") else {}),
                         "text": (read.get(s["id"]) or {}).get("text")} for s in secs]}
    md = markdown_of(doc)
    (d / "page.md").write_text(md, encoding="utf-8")
    pairs = lines_with_chunks(md)
    return {"markdown": md, "lines": [{"text": t, "section": c} for t, c in pairs]}


def found_in(key: str, value, kind: str | None, lines: list[str]) -> list[int]:
    """เลขบรรทัด (นับจาก 0) ที่ค่านี้ปรากฏ — ไว้ชี้ให้ดูเทียบ ไม่ได้บอกว่า rules หยิบจากบรรทัดไหน (หลายบรรทัดอาจมีค่าเดียวกัน)."""
    if value in (None, ""):
        return []
    try:
        if isinstance(value, date) or kind == "date":
            want = value if isinstance(value, date) else date.fromisoformat(str(value))
            hit = lambda ln: want in find_dates(ln)  # noqa: E731
        elif key == "vat_rate" or kind == "percent":
            want = Decimal(str(value)) * 100
            hit = lambda ln: want in find_percents(ln) or Decimal(str(value)) in find_numbers(ln)  # noqa: E731
        elif isinstance(value, Decimal) or kind == "money":
            want = Decimal(str(value))
            hit = lambda ln: want in find_numbers(ln)  # noqa: E731
        elif key.endswith("tax_id") or kind == "tax_id":
            hit = lambda ln: str(value) in re.sub(r"\D", "", ln)  # noqa: E731
        else:
            want = squash(str(value))
            hit = lambda ln: bool(want) and want in squash(ln)  # noqa: E731
    except (ValueError, InvalidOperation):
        return []
    return [i for i, ln in enumerate(lines) if hit(ln)][:6]


def stage_extract(job: dict, d: Path, cfg: dict) -> None:
    """ขั้น 7–9 จาก ``build_result`` ตัวเดียวกับคำสั่ง extract (rules ล้วน ไม่ใช้ LLM)."""
    md = job_stage(job, "text")["data"]["markdown"]
    ocr = job_stage(job, "ocr")["data"]
    pipe = FinanceOCRPipeline(cfg, engine=job["options"]["engine"], use_llm=False, with_reader=False)
    lines = pipe.lines_from_raw(md)
    confs = [r["confidence"] for r in ocr["sections"] if r.get("confidence") is not None]
    extra = [pipe._incomplete_check()] if any(r.get("truncated") for r in ocr["sections"]) else []
    res = pipe.build_result(job["file"], 1, lines, md, sum(confs) / len(confs) if confs else None, extra)
    data = res.data
    profile = pipe.profile_for(data)
    sq = squash("\n".join(lines))
    kinds = {DOC_PREFIX + f.key: f.type for f in profile.fields}
    labels = {DOC_PREFIX + f.key: f.label for f in profile.fields}
    flat = flatten(data)
    fields = [{"key": k, "label": labels.get(k), "value": str(v), "found": found_in(k, v, kinds.get(k), lines)} for k, v in flat.items()]
    out = json.loads(res.model_dump_json())
    (d / "result.json").write_text(res.model_dump_json(indent=2), encoding="utf-8")

    with state_lock:  # ขั้น 7–9 ปรากฏพร้อมกัน หน้าเว็บจะไม่เห็นผลสกัดที่ยังไม่มีระดับหลักฐาน
        set_stage(job, "extract", "done", {
            "profile": {"name": profile.name, "title": profile.title, "status": profile.status,
                        "matched": [a for a in profile.aliases if squash(a) and squash(a) in sq],
                        "required": list(profile.required), "primary_amount": profile.primary_amount,
                        "not_applicable": sorted(profile.not_applicable), "checks": list(profile.checks) if profile.checks else None,
                        "fields": [{"key": f.key, "label": f.label, "type": f.type} for f in profile.fields]},
            "document_type": data.document_type, "fields": fields, "items": out["data"]["items"], "remark": out["data"]["remark"],
            "lines": lines,
        })
        checks = res.checks
        primary = profile.primary_amount
        na = set(profile.not_applicable)
        has_primary = not primary or primary in na or flat.get(primary) not in (None, "")
        set_stage(job, "validate", "done", {
            "checks": out["checks"], "evidence": out["field_evidence"], "missing": res.missing_fields, "status": res.status,
            "rules": [
                {"rule": f"มียอดหลัก ({primary or 'ไม่กำหนด'})", "ok": has_primary, "fail_to": "failed"},
                {"rule": "ฟิลด์ที่ต้องมีครบ", "ok": not res.missing_fields, "fail_to": "needs_review"},
                {"rule": "ไม่มี check ที่ไม่ผ่าน", "ok": all(c.passed is not False for c in checks), "fail_to": "needs_review"},
                {"rule": "มี check ที่ผ่านอย่างน้อย 1 ข้อ", "ok": any(c.passed is True for c in checks), "fail_to": "needs_review"},
                {"rule": "โปรไฟล์ไม่ใช่ฉบับร่าง (draft)", "ok": profile.can_auto_approve, "fail_to": "needs_review"},
            ],
        })
        set_stage(job, "result", "done", {"json": out, "path": (d / "result.json").relative_to(ROOT).as_posix()})


# ---------------------------------------------------------------- ขั้นที่รันใน worker (โหลดโมเดล)

def run_worker(job: dict, d: Path, stage: str, args: list[str], on_event) -> None:
    cmd = [sys.executable, "-X", "utf8", str(WORKER), stage, str(d), *args]
    log = (d / f"worker_{stage}.log").open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log, text=True, encoding="utf-8", errors="replace", cwd=ROOT)
    with state_lock:
        job["proc"] = proc
    err = None
    for raw in proc.stdout:
        try:
            ev = json.loads(raw)
        except json.JSONDecodeError:
            continue  # ข้อความของ library ที่พิมพ์ลง stdout
        if ev.get("event") == "error":
            err = ev["error"]
        else:
            on_event(ev)
    proc.wait()
    log.close()
    with state_lock:
        job.pop("proc", None)
    if job.get("cancelled"):
        raise StageError("หยุดโดยผู้ใช้")
    if err or proc.returncode:
        tail = (d / f"worker_{stage}.log").read_text(encoding="utf-8", errors="replace").strip().splitlines()[-8:]
        raise StageError((err or f"worker จบด้วยรหัส {proc.returncode}") + ("\n" + "\n".join(tail) if tail else ""))


def stage_preprocess(job: dict, d: Path, cfg: dict) -> dict:
    out: dict = {}
    o = job["options"]["orientation"]

    def on_event(ev):
        if ev["event"] == "done":
            out.update({k: v for k, v in ev.items() if k != "event"})

    run_worker(job, d, "preprocess", [] if o == "auto" else ["--orientation", str(o)], on_event)
    return out


def stage_ocr(job: dict, d: Path, cfg: dict) -> dict:
    index = json.loads((d / "sections" / "index.json").read_text(encoding="utf-8"))["sections"]
    data = {"engine": job["options"]["engine"], "reader": None, "load_seconds": None, "total": len(index), "current": None, "sections": []}
    set_stage(job, "ocr", "running", data)

    def on_event(ev):
        with state_lock:
            if ev["event"] == "loaded":
                data.update(reader=ev["reader"], load_seconds=ev["seconds"])
            elif ev["event"] == "start":
                data["current"] = ev["id"]
            elif ev["event"] == "section":
                data["sections"].append({k: v for k, v in ev.items() if k != "event"})
                data["current"] = None
            job["rev"] += 1

    run_worker(job, d, "ocr", ["--engine", job["options"]["engine"]], on_event)
    return data


# ---------------------------------------------------------------- งาน

def job_stage(job: dict, key: str) -> dict:
    return next(s for s in job["stages"] if s["key"] == key)


def set_stage(job: dict, key: str, status: str, data: dict | None = None, **extra) -> None:
    with state_lock:
        s = job_stage(job, key)
        s["status"] = status
        if data is not None:
            s["data"] = data
        s.update(extra)
        job["rev"] += 1


RUNNERS = {"input": stage_input, "preprocess": stage_preprocess, "layout": stage_layout, "crops": stage_crops,
           "ocr": stage_ocr, "text": stage_text}


def run_job(job: dict) -> None:
    d = Path(job["dir"])
    cfg = load_config()
    try:
        for key, _title in STAGES:
            if key in ("validate", "result"):
                continue  # เขียนพร้อมกับ extract
            set_stage(job, key, "running", started=time.time())
            t = time.perf_counter()
            try:
                if key == "extract":
                    stage_extract(job, d, cfg)
                else:
                    set_stage(job, key, "done", RUNNERS[key](job, d, cfg))
            except StageError as e:
                set_stage(job, key, "error", error=str(e))
                return
            except Exception as e:  # noqa: BLE001 — แสดงบนหน้าเว็บ ไม่ให้เธรดตายเงียบ
                set_stage(job, key, "error", error=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")
                return
            set_stage(job, key, job_stage(job, key)["status"], seconds=round(time.perf_counter() - t, 2))
    finally:
        with state_lock:
            job["finished"] = True
            job["rev"] += 1
        run_lock.release()


def new_job(name: str, body: bytes, options: dict) -> dict:
    cfg = load_config()
    jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    d = DataPaths(cfg).root / "etc" / "demo" / jid
    d.mkdir(parents=True)
    (d / f"source{Path(name).suffix.lower()}").write_bytes(body)
    job = {"id": jid, "file": name, "dir": str(d), "options": options, "rev": 0, "finished": False, "created": time.time(),
           "stages": [{"key": k, "title": t, "status": "pending", "data": None} for k, t in STAGES]}
    (d / "options.json").write_text(json.dumps({"file": name, **options}, ensure_ascii=False, indent=1), encoding="utf-8")
    JOBS[jid] = job
    threading.Thread(target=run_job, args=(job,), daemon=True).start()
    return job


def public(job: dict) -> dict:
    return {k: v for k, v in job.items() if k not in ("proc", "dir")}


# ---------------------------------------------------------------- HTTP

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # ไม่พิมพ์ทุกคำขอ (มีชื่อไฟล์เอกสาร)
        pass

    def send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def json(self, obj, code: int = 200) -> None:
        self.send(code, json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            return self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/info":
            return self.json({"engines": ENGINES, "busy": run_lock.locked(), "exts": sorted(SUPPORTED_EXTS)})
        if m := re.fullmatch(r"/font/(Sarabun-(?:Regular|Bold)\.ttf)", path):
            return self.send(200, (FONTS / m.group(1)).read_bytes(), "font/ttf")
        if m := re.fullmatch(r"/api/job/([\w-]+)", path):
            job = JOBS.get(m.group(1))
            if not job:
                return self.json({"error": "ไม่พบงาน"}, 404)
            with state_lock:
                body = json.dumps(public(job), ensure_ascii=False, default=str)
            return self.send(200, body.encode("utf-8"), "application/json; charset=utf-8")
        if m := re.fullmatch(r"/job/([\w-]+)/(.+)", path):
            job = JOBS.get(m.group(1))
            if job:
                base = Path(job["dir"]).resolve()
                f = (base / unquote(m.group(2))).resolve()
                if f.is_file() and base in f.parents:  # กันพาธออกนอกโฟลเดอร์งาน
                    return self.send(200, f.read_bytes(), mimetypes.guess_type(f.name)[0] or "application/octet-stream")
        self.json({"error": "not found"}, 404)

    def do_POST(self):
        url = urlparse(self.path)
        if url.path == "/api/run":
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            name = Path(unquote(self.headers.get("X-Filename", "upload"))).name
            size = int(self.headers.get("Content-Length", 0))
            if Path(name).suffix.lower() not in SUPPORTED_EXTS:
                return self.json({"error": f"ไม่รองรับไฟล์ชนิดนี้ (ใช้ได้: {', '.join(sorted(SUPPORTED_EXTS))})"}, 400)
            if not 0 < size <= MAX_UPLOAD:
                return self.json({"error": f"ขนาดไฟล์ต้องไม่เกิน {MAX_UPLOAD // 2**20} MB"}, 400)
            body = self.rfile.read(size)
            engine = q.get("engine", "openthai")
            if engine not in ENGINES:
                return self.json({"error": "ตัวเลือกไม่ถูกต้อง"}, 400)
            if not run_lock.acquire(blocking=False):
                return self.json({"error": "มีงานกำลังรันอยู่ รอให้เสร็จก่อน (รันทีละงานเพื่อไม่ให้ GPU รับงานซ้อน)"}, 409)
            try:
                job = new_job(name, body, {"engine": engine, **FLOW_DEFAULTS})
            except Exception:
                run_lock.release()
                raise
            return self.json({"id": job["id"]})
        if m := re.fullmatch(r"/api/job/([\w-]+)/cancel", url.path):
            job = JOBS.get(m.group(1))
            if job and (proc := job.get("proc")):
                job["cancelled"] = True
                proc.kill()
            return self.json({"ok": True})
        self.json({"error": "not found"}, 404)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"เปิด http://127.0.0.1:{args.port}  (ผลแต่ละงานที่ data/etc/demo/) Ctrl+C เพื่อปิด", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
