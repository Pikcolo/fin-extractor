"""ชั้นตรวจความครบของ OCR รายส่วน (ขั้น ``ocr`` ของคำสั่ง ``sections`` หลังตัวอ่านหลัก)

ปัญหา: OpenThai (VLM) อ่านตารางได้ดี แต่บางภาพทำทั้งกล่องหาย (1306 s03 กล่องยอด, 1317–1319 s01 กล่องวันที่/เลขที่,
1310 s04 แถว VAT, 1308/1309 s01 แถว Invoice No.) ภาพเหล่านี้เป็นภาพกว้างที่มีสองกล่องวางข้างกันแล้วกล่องขวาหาย

1. **ตรวจ (CPU):** ตัวอ่านบรรทัด (OpenThai Text) อ่านภาพย่อยชุดเดียวกันซ้ำ เทียบกับตัวอ่านหลัก ค่าที่ตัวอ่านเสริมเห็นแต่ตัวอ่านหลักไม่เห็น:
   ยอดเงิน · วันที่ · รหัสยาว (เช่นเลขที่ใบกำกับ) ภาพที่มีค่าแบบนี้ = อาจอ่านไม่ครบ
2. **แยกซ้าย/ขวาแล้วอ่านใหม่ (GPU เฉพาะภาพที่ถูกชี้):** แยกที่แนวว่างตลอดความสูงที่ใกล้กึ่งกลาง (``split_column``) ให้ตัวอ่านหลักอ่านสองครึ่ง
   **รับผลเมื่อค่าที่ยังขาดลดลงเท่านั้น** ภาพที่ถูกชี้เพราะตัวอ่านเสริมอ่านเพี้ยนเอง ครึ่งที่แยกก็ไม่มีค่านั้น จึงคงข้อความเดิม
3. **แนบข้อความเสริม:** ภาพที่ยังขาด **ยอดเงินหรือวันที่** หลังขั้น 2 ได้ข้อความของตัวอ่านเสริมต่อท้าย .md
   (``<!-- sNN kind supplement openthai-text -->``) หลังท่อนหลักทุกท่อน ค่าจากตัวอ่านหลักจึงชนะเมื่อมีทั้งคู่
   รหัสไม่ใช้ตัดสินขั้นนี้ เพราะตัวอ่านเสริมอ่าน O/0 I/1 เพี้ยนบ่อย (ค่าเพี้ยนจะกลายเป็นเลขที่เอกสารผิด)

ผลวัด (OpenThai regions 18 ใบ 2026-11-25, docs/04-openthai-ocr.md §7): ข้อความเสริมอย่างเดียว ยอดเงินถูก 27 → 33 เลขที่ 5 → 7 วันที่ 10 → 12
แยกซ้าย/ขวา (ภาพเป้าหมาย 8 ภาพ) ยอดเงิน 34 เลขที่ 10 วันที่ 13 ไม่มีค่าผิดเพิ่ม
"""

from __future__ import annotations

import gc
import json
import re
import time
from pathlib import Path

import numpy as np

from ..extraction.patterns import find_amounts, find_dates, fix_ocr_numbers
from ..io import read_image
from ..paths import DataPaths, doc_dir
from ..readers import TORCH_ENGINES, make_reader
from ..readers.lines import normalize_thai
from .read import (
    coverage_file,
    joined_halves,
    split_file,
    clean_index,
    fingerprint,
    image_locator,
    images_dir,
    layout_file,
    write_markdown,
)

# ตัวอักษร O ที่อยู่ติดตัวเลข = เลข 0 ที่ตัวอ่านบรรทัดอ่านผิด ("2,O00.O0" 1306, "ST6910-O036" 1317)
_O_DIGIT = re.compile(r"(?<=[\d,.])[Oo]|[Oo](?=[\d,.]\d)")
_CODE_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9][A-Za-z0-9\-/]*[A-Za-z0-9]")
# รหัสยาวอย่างน้อยเท่าเลขที่เอกสารที่สั้นที่สุดใน 18 ใบ (ST6910-0036 = 10 ตัวเมื่อตัดขีด) ต้องมีตัวเลขในรูปเดิม (ไม่ใช่คำอย่าง COLLECTOR)
MIN_CODE_LEN = 10
_LOOKALIKE = str.maketrans("OIL", "011")  # ตัวที่ OCR สับสนกับตัวเลข เทียบรหัสสองตัวอ่านในรูปเดียวกัน


def normalize_digits(text: str) -> str:
    return _O_DIGIT.sub("0", fix_ocr_numbers(normalize_thai(re.sub(r"<[^>]+>", " ", text or ""))))


def _codes(text: str) -> set[str]:
    out = set()
    for c in _CODE_RE.findall(normalize_thai(re.sub(r"<[^>]+>", " ", text or ""))):
        k = re.sub(r"[-/]", "", c.upper())
        if len(k) >= MIN_CODE_LEN and re.search(r"\d", c):
            out.add(k.translate(_LOOKALIKE))
    return out


def missing_values(primary: str, check: str) -> dict[str, list[str]]:
    """ยอดเงิน/วันที่/รหัสยาวที่อยู่ใน ``check`` (ตัวอ่านเสริม) แต่ไม่อยู่ใน ``primary`` (ตัวอ่านหลัก)."""
    a, b = normalize_digits(check), normalize_digits(primary)
    money = sorted({str(x) for x in find_amounts(a)} - {str(x) for x in find_amounts(b)})
    dates = sorted({d.isoformat() for d in find_dates(a)} - {d.isoformat() for d in find_dates(b)})
    return {"money": money, "dates": dates, "codes": sorted(_codes(check) - _codes(primary))}


def _squash(text: str) -> str:
    """รูปเทียบคำ: ตัดช่องว่าง/เครื่องหมายคั่น ตัวพิมพ์ใหญ่ แปลง O/I/L ให้เหมือนกันทั้งสองตัวอ่าน (กันนับเป็นไม่พบเพราะอ่าน 0/O ต่างกัน)."""
    return re.sub(r"[\s|:：.,()\-/#*_]", "", normalize_digits(text)).upper().translate(_LOOKALIKE)


def word_coverage(check_lines: list[dict], primary: str) -> list[dict]:
    """คำของตัวอ่านเสริม (กรอบ + ข้อความ จาก ``LineOCR.read()["lines"][].words``) → พบในข้อความของตัวอ่านหลักไหม
    ``found`` = None สำหรับคำตัวเดียว (สั้นเกินกว่าจะบอกได้) ใช้วาดบนภาพ: ส่วนที่ตัวอ่านหลักไม่ได้อ่านเป็นกลุ่มคำที่ไม่พบ."""
    target = _squash(primary)
    out = []
    for ln in check_lines:
        for w in ln.get("words") or [{"text": ln.get("text", ""), "box": ln["box"]}]:
            t = _squash(w["text"])
            if not t:
                continue
            xs, ys = [p[0] for p in w["box"]], [p[1] for p in w["box"]]
            out.append({"box": [round(min(xs)), round(min(ys)), round(max(xs) - min(xs)), round(max(ys) - min(ys))],
                        "text": w["text"], "found": (t in target) if len(t) >= 2 else None})
    return out


def n_missing(m: dict[str, list[str]]) -> int:
    return len(m["money"]) + len(m["dates"]) + len(m["codes"])


def split_column(img: np.ndarray, rcfg: dict) -> int | None:
    """จุดแยกซ้าย/ขวาของภาพย่อย: แนวว่างตลอดความสูง (ไม่นับเส้นตั้ง) ที่ใกล้กึ่งกลางภาพที่สุด
    ไม่มีเลย = ไม่นับเส้นนอนยาวด้วย (1306: เส้นขอบล่างยาวตลอดภาพบังทุกคอลัมน์) ยังไม่มี = None
    ใช้ตัวตรวจเส้นและค่าใน configs/regions/pipeline.yaml (``lines``) ภาพย่อยกว้างใกล้เคียงหน้าจึงใช้ค่าเดียวกันได้
    ลองแบบ "แนวว่างที่มีเส้นตั้งยาวสุด" แล้วแย่กว่า (1306 แยกที่ขอบขวา, 1319 เส้นขอบกล่องจางหาไม่เจอ)."""
    from .regions import lines

    ink = lines.ink_mask(img).copy()
    for v in lines.detect_vlines(img, rcfg["lines"]["v_min_len_frac"]):
        x = int(v["x"])
        ink[int(v["y0"]):int(v["y1"]) + 1, max(0, x - 4):x + 5] = 0
    runs = _blank_runs(ink)
    if not runs:
        for h in lines.detect_hlines(img, rcfg["lines"]["min_len_frac"]):
            y = int(h["y"])
            ink[max(0, y - 4):y + 5, int(h["x0"]):int(h["x1"]) + 1] = 0
        runs = _blank_runs(ink)
    if not runs:
        return None
    w = ink.shape[1]
    s, e = min(runs, key=lambda r: abs((r[0] + r[1]) / 2 - w / 2))
    return (s + e) // 2


def _blank_runs(ink: np.ndarray) -> list[tuple[int, int]]:
    """ช่วงคอลัมน์ที่ไม่มีหมึกเลยตลอดความสูง ไม่นับช่วงที่ติดขอบภาพ."""
    blank = ink.sum(axis=0) == 0
    w = len(blank)
    runs, x = [], 0
    while x < w:
        if blank[x]:
            s = x
            while x < w and blank[x]:
                x += 1
            if s > 0 and x < w:
                runs.append((s, x))
        x += 1
    return runs


def _load(path: Path, layout: list[dict], redo: bool) -> dict | None:
    if redo or not path.exists():
        return None
    prev = json.loads(path.read_text(encoding="utf-8"))
    return prev if prev.get("layout") == fingerprint(layout) else None


def check_dataset(paths: DataPaths, cfg: dict, dataset: str, stems: list[str], engine: str, variant: str,
                  redo: bool = False) -> list[dict]:
    """ขั้น 1–3 ของชั้นตรวจกับ ``stems`` (ผลของตัวอ่านหลัก ``engine``) แล้วเขียน .md ใหม่
    ใช้ผลเดิม (``.coverage.json`` ตัวอ่านเสริม · ``.split.json`` ครึ่งที่อ่านใหม่) ถ้าผังไม่เปลี่ยน โหลดตัวอ่านหลักบน GPU เฉพาะเมื่อมีภาพต้องแยกใหม่
    ข้ามเมื่อ ``engine`` คือตัวอ่านเสริมเอง หรือภาพย่อยเป็น variant=color (ไม่มีไฟล์ภาพ)."""
    cc = cfg.get("coverage_check") or {}
    check_engine = cc.get("engine", "openthai-text")
    if not cc.get("enabled", True) or engine == check_engine or variant == "color":
        return []
    from .regions.run import RegionConfig

    rcfg = RegionConfig().cfg
    raw_dir = paths.raw_data(engine, dataset, variant)
    checker = primary = None
    summary = []
    try:
        for stem in stems:
            src = doc_dir(raw_dir, stem, nest=True) / f"{stem}_p1.sections.json"
            if not src.exists():
                continue
            layout = json.loads(layout_file(paths, dataset, stem, variant).read_text(encoding="utf-8"))["layout_sections"]
            first = {s["id"]: s.get("text") or "" for s in json.loads(src.read_text(encoding="utf-8"))["sections"]}
            clean_dir = images_dir(paths, dataset, stem, variant)
            entries = clean_index(clean_dir, layout, stem)
            image = lambda sid: read_image(clean_dir / entries[sid]["file"])  # noqa: E731

            prev = _load(coverage_file(src), layout, redo)
            if prev:
                texts = {s["id"]: s["text"] for s in prev["sections"]}
            else:
                if checker is None:
                    try:
                        checker = make_reader(check_engine, cfg)
                        checker._load()
                    except Exception as e:  # noqa: BLE001 — ไม่มีโมเดล/onnxruntime: ข้ามชั้นตรวจ ไม่ทำให้ flow หลักล้ม
                        print(f"ข้ามชั้นตรวจความครบ ({check_engine} ใช้ไม่ได้: {type(e).__name__}: {str(e)[:120]})", flush=True)
                        return summary
                t0 = time.perf_counter()
                texts = {s["id"]: checker.recognize(image(s["id"])).text for s in layout
                         if s["keep"] and entries.get(s["id"], {}).get("file") and first.get(s["id"])}
                print(f"    {stem[-4:]} ตรวจความครบ {len(texts)} ส่วน {time.perf_counter() - t0:.1f}s (CPU)", flush=True)

            before = {sid: missing_values(first.get(sid, ""), t) for sid, t in texts.items()}
            splits = (_load(split_file(src), layout, redo) or {}).get("sections", {})
            todo = [sid for sid, m in before.items() if n_missing(m) and sid not in splits]
            for sid in todo:
                img = image(sid)
                x = split_column(img, rcfg)
                if x is None:
                    splits[sid] = {"split": None}
                    continue
                if primary is None:
                    gc.collect()
                    t0 = time.perf_counter()
                    primary = make_reader(engine, cfg)
                    if hasattr(primary, "_load"):
                        primary._load()
                    print(f"    โหลดตัวอ่านหลัก ({engine}) อ่านครึ่งซ้าย/ขวา {time.perf_counter() - t0:.1f}s", flush=True)
                t0 = time.perf_counter()
                r = {"split": x, "left": primary.recognize(img[:, :x]).text, "right": primary.recognize(img[:, x:]).text}
                after = missing_values(joined_halves(r), texts[sid])
                r.update(missing_before=n_missing(before[sid]), missing_after=n_missing(after),
                         accepted=n_missing(after) < n_missing(before[sid]), seconds=round(time.perf_counter() - t0, 1))
                splits[sid] = r
                if engine in TORCH_ENGINES:
                    import torch

                    torch.cuda.empty_cache()
                print(f"    {stem[-4:]} {sid} แยกที่ x={x} ขาด {r['missing_before']} → {r['missing_after']} "
                      f"{'ใช้ผลที่แยก' if r['accepted'] else 'คงข้อความเดิม'} {r['seconds']}s", flush=True)
            split_file(src).write_text(json.dumps({"source": stem, "engine": engine, "layout": fingerprint(layout), "sections": splits},
                                                  ensure_ascii=False, indent=1), encoding="utf-8")

            effective = {sid: joined_halves(splits[sid]) if splits.get(sid, {}).get("accepted") else first.get(sid, "") for sid in texts}
            rows = [{"id": sid, "text": t, **missing_values(effective[sid], t)} for sid, t in texts.items()]
            flagged = [r["id"] for r in rows if r["money"] or r["dates"]]
            coverage_file(src).write_text(json.dumps({"source": stem, "engine": check_engine, "against": engine,
                                                      "layout": fingerprint(layout), "flagged": flagged,
                                                      "split": [sid for sid, r in splits.items() if r.get("accepted")], "sections": rows},
                                                     ensure_ascii=False, indent=1), encoding="utf-8")
            write_markdown(src, image_locator(paths, dataset, stem, variant), layout)
            summary.append({"file": stem[-4:], "flagged": flagged, "split": [sid for sid, r in splits.items() if r.get("accepted")]})
            if flagged:
                print(f"    {stem[-4:]} ยังขาด (แนบข้อความเสริม): " + " · ".join(
                    f"{r['id']} " + " ".join(r["money"] + r["dates"]) for r in rows if r["id"] in flagged), flush=True)
    finally:
        del primary
        gc.collect()
    return summary
