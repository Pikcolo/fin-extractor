"""Router ก่อน OCR: อ่านภาพย่อยทุกส่วนด้วย OpenThai (คัดกรอง) แล้วตัดสินว่าส่วนไหนเกี่ยวกับข้อมูลการเงิน

    python -X utf8 scripts/route_sections.py data/raw/finance_doc_scan            # อ่าน (GPU) แล้วตัดสินใจ
    python -X utf8 scripts/route_sections.py data/raw/finance_doc_scan --decide   # ตัดสินใจใหม่จากข้อความที่อ่านไว้ (ไม่ใช้ GPU)
    python -X utf8 scripts/route_sections.py data/raw/finance_doc_scan --second-opinion   # ให้ OXR อ่านส่วนที่กำลังจะถูกตัดทิ้ง (GPU)

ความเห็นที่สอง (--second-opinion): ตัวอ่านคัดกรองภาษาไทยอ่านเพี้ยนได้ (เจอจริง: ตัวอักษรเทาจางบนพื้นฟ้าของ 1317 ถูกอ่านเป็นประโยคมั่ว) และ
router ตัดสินจากข้อความนั้น จึงให้ OXR (อ่านตัวเลขได้ดี ไม่รู้ไทย) ลองอ่านเฉพาะส่วนที่ **กำลังจะถูกตัดทิ้ง** ทั้งแบบ luma และ flat ถ้าเจอค่าการเงิน
กลับมาเก็บ ผลเก็บไว้ในไฟล์ผล router (``oxr_text`` / ``oxr_values``) ``--decide`` จึงใช้ซ้ำได้โดยไม่ต้องใช้ GPU
ขั้นอ่านกับขั้นตัดสินใจแยกกัน: อ่านครั้งเดียวเก็บข้อความไว้ที่ ``data/preprocess/<dataset>/<profile>/sections/router/<ไฟล์>.json`` แล้วปรับเกณฑ์
ใน ``fin_extractor/routing/relevance.py`` ได้โดยไม่ต้องอ่านซ้ำ (``--decide``) ผลมีภาพ ``<ไฟล์>_overlay.png`` (เขียว = เก็บ แดง = ตัดทิ้ง)
ให้ตรวจด้วยตา อินพุตคือภาพย่อย clean/luma/<ไฟล์>/sNN.png (สร้างด้วย make_clean_sections.py)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notebooks" / "preprocess_lab"))

import labkit as L  # noqa: E402
from fin_extractor.config import load_config  # noqa: E402
from fin_extractor.paths import dataset_of  # noqa: E402
from fin_extractor.extraction import markdown_to_lines  # noqa: E402
from fin_extractor.pipeline import FinanceOCRPipeline  # noqa: E402
from fin_extractor.routing import build_aliases, route, signals  # noqa: E402
from fin_extractor.validation.evidence import flatten  # noqa: E402

COLORS = {"data": (0, 160, 0), "contrib": (0, 130, 0), "short": (200, 160, 0), "label": (200, 120, 0), "small": (200, 120, 120), "companion": (0, 170, 170),
          "residual": (170, 100, 170), "drop": (0, 0, 220), "duplicate": (150, 150, 150)}


def read_all(stem: str, dataset: str, reader) -> dict:
    d = L.PATHS.section_clean(dataset, stem, "luma")
    index = json.loads((d / "index.json").read_text(encoding="utf-8"))["sections"]
    rows = []
    for s in index:
        img = cv2.imdecode(np.fromfile(str(d / s["file"]), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        t = time.perf_counter()
        try:
            res = reader.recognize(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
            text, trunc, err = res.markdown, res.truncated, None
        except Exception as e:  # noqa: BLE001
            text, trunc, err = "", False, repr(e)[:200]
        rows.append({"id": s["id"], "kind": s["kind"], "bbox": s["bbox"], "table": s["table"], "size": s["size"],
                     "text": text, "truncated": trunc, "error": err, "seconds": round(time.perf_counter() - t, 2)})
    return {"source": stem, "sections": rows}


def contributions(secs: list[dict], pipe) -> dict[str, list[str]]:
    """ฟิลด์ที่ rules สกัดได้จากข้อความของแต่ละส่วน (ตัวสกัดจริงเป็นผู้ตัดสินว่าส่วนนี้ช่วยอะไร)."""
    out = {}
    for s in secs:
        try:
            keys = [k for k in flatten(pipe.extract(markdown_to_lines(s["text"]), s["text"])) if k != "document_type" and not k.startswith("remark[")]
        except Exception:  # noqa: BLE001 — ข้อความมั่วทำให้สกัดล้มได้ ถือว่าไม่มีฟิลด์
            keys = []
        out[s["id"]] = keys
    return out


def decide(doc: dict, stem: str, dataset: str, aliases, pipe) -> None:
    full = L.load_cached_preprocessed(stem, dataset=dataset)
    work, _ = L._resize_to_work(full)
    secs = doc["sections"]
    decisions = route(secs, {s["id"]: s["text"] for s in secs}, aliases, page_size=(work.shape[1], work.shape[0]),
                      contrib=contributions(secs, pipe))
    for s, dec in zip(secs, decisions):
        s.update(keep=dec.keep, role=dec.role, reason=dec.reason, labels=dec.labels, values=dec.values)
    apply_second_opinion(doc)  # ผลของ OXR ที่เก็บไว้ (ถ้ามี) ใช้ก่อนวาดภาพและบันทึก
    over = work.copy()
    for s in sorted(secs, key=lambda s: s["role"] not in ("duplicate", "residual")):  # ซ้ำ/กล่องครอบวาดก่อน ไม่ให้ทับส่วนจริงที่ bbox เดียวกัน
        x, y, w, h = s["bbox"]
        cv2.rectangle(over, (x, y), (x + w, y + h), COLORS[s["role"]], 2 if s["keep"] else 4)
        cv2.putText(over, s["id"][1:], (x + 3, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 0, 90), 1, cv2.LINE_AA)
    out = L.PATHS.section_router(dataset)
    cv2.imencode(".png", over)[1].tofile(str(out / f"{stem}_overlay.png"))
    (out / f"{stem}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")


def apply_second_opinion(doc: dict) -> None:
    """ส่วนที่ router ตัดทิ้งแต่ OXR เจอค่าการเงิน → กลับมาเก็บ (ใช้ผลที่เก็บไว้ในไฟล์ ไม่ใช้ GPU)."""
    for s in doc["sections"]:
        if s["role"] == "drop" and s.get("oxr_values"):
            s.update(keep=True, role="data", reason="OXR (ความเห็นที่สอง) พบค่าการเงิน: " + ", ".join(s["oxr_values"]), values=s["oxr_values"])


def second_opinion(doc: dict, stem: str, dataset: str, oxr) -> int:
    """ให้ OXR อ่านส่วนที่ตัดทิ้งทั้ง luma และ flat เก็บข้อความ/ค่าที่พบ คืนจำนวนส่วนที่อ่าน."""
    n = 0
    for s in doc["sections"]:
        if s["role"] != "drop" or "oxr_values" in s:
            continue
        texts, values = [], set()
        for variant in ("luma", "flat"):
            img = cv2.imdecode(np.fromfile(str(L.PATHS.section_clean(dataset, stem, variant) / f"{s['id']}.png"), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            try:
                t = oxr.recognize(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)).markdown
            except Exception as e:  # noqa: BLE001
                t = ""
                print(f"    OXR ล้ม {stem[-4:]} {s['id']} {variant}: {type(e).__name__}", flush=True)
            texts.append(t)
            values |= set(signals(t)["values"])
        s["oxr_text"], s["oxr_values"] = texts, sorted(values)
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--only", nargs="*", help="เลขท้ายชื่อไฟล์")
    ap.add_argument("--decide", action="store_true", help="ตัดสินใจใหม่จากข้อความที่อ่านไว้ (ไม่อ่านใหม่ ไม่ใช้ GPU)")
    ap.add_argument("--redo", action="store_true", help="อ่านใหม่แม้มีข้อความเก็บไว้แล้ว")
    ap.add_argument("--second-opinion", action="store_true", help="ให้ OXR อ่านส่วนที่กำลังจะถูกตัดทิ้ง (ใช้ GPU) แล้วกลับมาเก็บถ้าเจอค่าการเงิน")
    args = ap.parse_args()
    cfg = load_config()
    dataset = dataset_of(args.input)
    aliases = build_aliases(cfg)
    pipe = FinanceOCRPipeline(cfg, mode="vlm", engine="openthai", use_llm=False, with_reader=False)  # ตัวสกัด rules ล้วน (ไม่โหลดโมเดล)
    out = L.PATHS.section_router(dataset)
    out.mkdir(parents=True, exist_ok=True)
    stems = sorted(p.name for p in L.PATHS.section_clean(dataset, variant='luma').iterdir() if p.is_dir())
    if args.only:
        stems = [s for s in stems if any(s.endswith(o) for o in args.only)]
    reader = oxr = None
    for stem in stems:
        f = out / f"{stem}.json"
        t0 = time.perf_counter()
        if f.exists() and not args.redo:
            doc = json.loads(f.read_text(encoding="utf-8"))
            how = "ใช้ข้อความที่อ่านไว้"
        elif args.decide:
            print(f"{stem[-4:]}: ยังไม่เคยอ่าน (ข้าม)", flush=True)
            continue
        else:
            if reader is None:
                from fin_extractor.vlm.openthai_ocr import OpenThaiOCR

                reader = OpenThaiOCR(cfg["openthai"])
            doc = read_all(stem, dataset, reader)
            how = "อ่านใหม่"
        decide(doc, stem, dataset, aliases, pipe)
        if args.second_opinion:
            if oxr is None:
                from fin_extractor.vlm.oxr_local import OXRLocal

                oxr = OXRLocal({**cfg["oxr_local"], "max_new_tokens": 96})  # ตัวเลขสั้น ช่องไทยที่วนซ้ำจะได้ไม่เสียเวลา
            n = second_opinion(doc, stem, dataset, oxr)
            print(f"    ความเห็นที่สอง (OXR) อ่าน {n} ส่วนที่กำลังถูกตัด", flush=True)
            decide(doc, stem, dataset, aliases, pipe)  # ตัดสินใหม่เพื่อใช้ผลของ OXR + วาดภาพใหม่
        kept = sum(s["keep"] for s in doc["sections"])
        roles = {r: sum(s["role"] == r for s in doc["sections"]) for r in COLORS}
        print(f"{stem[-4:]}: {len(doc['sections'])} ส่วน เก็บ {kept} ตัดทิ้ง {len(doc['sections']) - kept} {roles} · {how} {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
