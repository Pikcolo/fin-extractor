"""ขั้น 4 ของ flow: OCR ทีละภาพ section แล้วเก็บข้อความ + ที่มาของแต่ละท่อน (04_ocr/<dataset>/<engine>/<วิธี>/<ไฟล์>/)

ไฟล์ที่เขียนต่อหนึ่งเอกสาร (ชื่อ <ไฟล์>_p1.* เหมือนกันทุกขั้น)
    .sections.json  ผลดิบรายส่วน (ข้อความ เวลา ธง truncated ลายนิ้วมือของผัง ใช้ตัดสินว่าข้าม/ต่อ/ทำใหม่)
    .md             ข้อความทุกส่วนต่อกันตามลำดับอ่าน <!-- sNN kind --> นำหน้าแต่ละท่อน — ไฟล์นี้เท่านั้นที่ extract อ่าน
    .chunks.json    ที่มาของแต่ละท่อน: ภาพ section ที่ส่งเข้า OCR + ข้อความที่ได้ + ช่วงบรรทัดใน .md (ย้ายมาจาก scripts/sections/run_sections.py)
GPU ใช้เฉพาะ read_dataset
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from ..paths import DataPaths, doc_dir
from .crops import load_final_image
from .layout import crop_full, resize_to_work

ENGINES = ("openthai", "typhoon", "paddle", "oxr")  # ทุก engine อ่านทีละ section เท่านั้น (ไม่มีทางอ่านทั้งหน้าใน CLI)
TORCH_ENGINES = ("openthai", "oxr")  # ตัวที่ใช้ torch ในเครื่อง (typhoon เรียก Ollama, paddle ห้ามอยู่ร่วม process กับ torch)


def kept_layout(layout: list[dict]) -> list[tuple[str, list[int]]]:
    return [(s["id"], list(s["bbox"])) for s in layout if s["keep"]]


def fingerprint(layout: list[dict]) -> list[list]:
    """ส่วนที่เก็บของผังปัจจุบัน (id + bbox) บันทึกไว้ในผลทุกไฟล์ ใช้ตัดสินว่าผลเก่าข้ามหรือต่อได้ไหม — ผังเปลี่ยนแล้วผลเก่าไม่ตรงอีกต่อไป."""
    return [[i, b] for i, b in kept_layout(layout)]


def clean_index(clean_dir: Path, layout: list[dict], stem: str) -> dict[str, dict]:
    """index.json ของภาพย่อยใน 03_sections/…/images/<วิธี>/ (id → รายการ) หยุดถ้าไม่ตรงกับผังปัจจุบัน — ภาพล้าสมัยจะอ่านผิดส่วนโดยไม่รู้ตัว."""
    path = clean_dir / "index.json"
    if not path.exists():
        sys.exit(f"{stem[-4:]}: ไม่มีภาพย่อยใน {clean_dir} — รัน: python -m fin_extractor sections <input> --from images")
    index = json.loads(path.read_text(encoding="utf-8"))["sections"]
    if [(s["id"], list(s["bbox"])) for s in index] != kept_layout(layout):
        sys.exit(f"{stem[-4:]}: ภาพย่อยใน {clean_dir} ไม่ตรงกับผังแบ่งปัจจุบัน (สร้างจากผังเก่า) — รัน: python -m fin_extractor sections <input> --from images --redo")
    return {s["id"]: s for s in index}


def read_clean_image(clean_dir: Path, entry: dict, variant: str) -> np.ndarray:
    """ภาพเทาช่องเดียวจากไฟล์ → BGR."""
    gray = cv2.imdecode(np.fromfile(str(clean_dir / entry["file"]), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class PaddleSectionReader:
    """PaddleOCR ในรูปเดียวกับตัวอ่าน VLM: ข้อความของภาพ section เป็นบรรทัด (ไม่มีตาราง Markdown) ``finish_reason`` เป็น None เสมอ."""

    name = "paddle"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.engine = None

    def _load(self):
        if self.engine is None:
            from ..ocr import create_engine

            self.engine = create_engine("paddle", self.cfg)
        return self.engine

    def recognize(self, img: np.ndarray):
        from ..vlm.base import VLMResult

        return VLMResult(self.name, self._load().recognize(img).text)


def make_reader(engine: str, cfg: dict):
    """ตัวอ่านของ engine นั้น (import ตอนใช้ ไม่โหลด torch/โมเดลของตัวที่ไม่ได้เลือก)."""
    if engine == "oxr":
        from ..vlm.oxr_local import OXRLocal
        return OXRLocal(cfg["oxr_local"])
    if engine == "typhoon":
        from ..vlm.typhoon_ocr import TyphoonOCR
        return TyphoonOCR(cfg["vlm"])
    if engine == "paddle":
        return PaddleSectionReader(cfg)
    from ..vlm.openthai_ocr import OpenThaiOCR
    return OpenThaiOCR(cfg["openthai"])


def markdown_blocks(doc: dict) -> list[tuple[dict, str]]:
    """(ส่วน, บล็อกข้อความ) ของทุกส่วนที่อ่านแล้วตามลำดับอ่าน บล็อกขึ้นต้นด้วยคอมเมนต์ HTML บอกว่ามาจากส่วนไหน (rules ตัดคอมเมนต์ทิ้ง)."""
    blocks = []
    for s in doc["sections"]:
        if not s.get("text"):
            continue
        part = f" {s['part']}/{s['parts']}" if s.get("part") else ""
        blocks.append((s, f"<!-- {s['id']} {s['kind']}{part} -->\n{s['text'].strip()}"))
    return blocks


def markdown_of(doc: dict) -> str:
    """ข้อความของทุกส่วนที่อ่านแล้ว ต่อกันตามลำดับอ่าน คั่นด้วยบรรทัดว่าง มีคอมเมนต์ HTML บอกว่าแต่ละท่อนมาจากส่วนไหน."""
    return "\n\n".join(b for _s, b in markdown_blocks(doc)) + "\n"


def script_mix(text: str) -> dict[str, int]:
    """จำนวนอักขระไทย / ละติน / ตัวเลข ในข้อความที่ **ตัวอ่านตอบมา** (นับจากผล ไม่ใช่สิ่งที่โมเดลเลือก — ดู docs/12 หัวข้อ "เลือกภาษา")
    ตัดแท็ก HTML ของตารางก่อนนับ."""
    import re
    t = re.sub(r"<[^>]+>", " ", text)
    return {"thai": sum("฀" <= c <= "๿" for c in t),
            "latin": sum(c.isascii() and c.isalpha() for c in t),
            "digit": sum(c.isdigit() for c in t)}


def chunks_of(doc: dict, image_of) -> dict:
    """หมายเหตุของข้อความดิบ: ท่อนข้อความแต่ละท่อนใน .md มาจากภาพ section ไหน
    ``image_of(section)`` คืนที่อยู่ภาพที่ส่งเข้าตัวอ่าน (หรือ None) บรรทัดนับจาก 1 ตามไฟล์ .md (รวมบรรทัดคอมเมนต์)."""
    chunks, line = [], 1
    for s, block in markdown_blocks(doc):
        n = block.count("\n") + 1
        chunks.append({"section": s["id"], "kind": s["kind"], **({"part": s["part"], "parts": s["parts"]} if s.get("part") else {}),
                       "md_lines": [line, line + n - 1], "page_bbox_xywh": s["bbox"], "image": image_of(s), "text": s["text"].strip(),
                       "chars": len(s["text"].strip()), "script_mix": script_mix(s["text"]),
                       **({"truncated": True} if s.get("truncated") else {}), **({"seconds": s["seconds"]} if "seconds" in s else {})})
        line += n + 1  # + บรรทัดว่างคั่นบล็อก
    not_read = [{"section": s["id"], "kind": s["kind"], "reason": s.get("skipped") or ("error" if "error" in s else "no text")}
                for s in doc["sections"] if not s.get("text")]
    return {"source": doc["source"], "engine": doc.get("engine"), "variant": doc.get("variant"),
            "note": "text คือข้อความที่ได้จากภาพนั้น (ชุดเดียวกับในท่อนของ .md) · page_bbox_xywh อยู่บนสำเนาด้านยาว 1600 px · md_lines คือบรรทัดของท่อนนั้นใน .md · image คือภาพที่ส่งเข้าตัวอ่าน (variant=color ไม่มีไฟล์ ตัดจาก final ตาม page_bbox_xywh) · "
                    "script_mix นับจากข้อความที่ตัวอ่านตอบ OpenThai ไม่มีขั้นเลือกภาษา (โมเดลเดียวอ่านทุกภาษา)",
            "chunks": chunks, "not_read": not_read}


def write_markdown(sections_json: Path, image_of=lambda s: None) -> Path:
    """เขียน .md (ข้อความดิบที่ extract อ่าน) และ .chunks.json (ที่มาของแต่ละท่อน ไม่ถูกอ่านโดย extract)."""
    stem = sections_json.name.removesuffix(".sections.json")
    doc = json.loads(sections_json.read_text(encoding="utf-8"))
    out = sections_json.with_name(stem + ".md")
    out.write_text(markdown_of(doc), encoding="utf-8")
    sections_json.with_name(stem + ".chunks.json").write_text(json.dumps(chunks_of(doc, image_of), ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def image_locator(paths: DataPaths, dataset: str, stem: str, variant: str):
    """ฟังก์ชันที่บอกที่อยู่ภาพของ section (สัมพัทธ์กับ data_dir) ใช้ใน .chunks.json."""
    if variant == "color":
        return lambda s: None
    entries = {e["id"]: e for e in json.loads((paths.section_images(dataset, stem, variant) / "index.json").read_text(encoding="utf-8"))["sections"]}
    base = paths.section_images(dataset, stem, variant)
    data_dir = paths.root

    def locate(s: dict):
        f = entries.get(s["id"], {}).get("file")
        return (base / f).relative_to(data_dir).as_posix() if f else None
    return locate


def gpu_status() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,temperature.gpu", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return out or "?"
    except Exception:  # noqa: BLE001
        return "?"


def _is_done(sections_json: Path, layout_json: Path) -> bool:
    """ผล OCR ของไฟล์นี้ครบและตรงกับผังปัจจุบัน (ใช้ข้ามโดยไม่ต้องโหลดโมเดล)."""
    if not sections_json.exists():
        return False
    prev = json.loads(sections_json.read_text(encoding="utf-8"))
    layout = json.loads(layout_json.read_text(encoding="utf-8"))["layout_sections"]
    return bool(prev.get("complete")) and prev.get("layout") == fingerprint(layout)


def read_dataset(paths: DataPaths, cfg: dict, dataset: str, stems: list[str], engine: str, variant: str,
                 redo: bool = False, markdown_only: bool = False) -> list[dict]:
    """ขั้น 4: OCR ทีละภาพ section ของ ``stems`` เก็บข้อความ + ที่มา (``.sections.json`` ``.md`` ``.chunks.json``) ที่ 04_ocr/<dataset>/<engine>/<วิธี>/<ไฟล์>/
    คืนสรุปต่อไฟล์ที่รันรอบนี้ ใช้ GPU เฉพาะส่วนนี้ (``markdown_only`` = สร้าง .md/.chunks.json จากผลที่มี ไม่โหลดโมเดล)."""
    layout_dir = paths.section_layout(dataset)
    raw_dir = paths.raw_data(engine, dataset, variant)  # variant: luma | flat | bin = ไฟล์ใน images/<วิธี>/ · color = ตัดจากภาพ final สีในหน่วยความจำ
    if markdown_only:
        for stem in stems:
            src = doc_dir(raw_dir, stem, nest=True) / f"{stem}_p1.sections.json"
            if src.exists():
                print(f"{stem[-4:]}: {write_markdown(src, image_locator(paths, dataset, stem, variant))}", flush=True)
            else:
                print(f"{stem[-4:]}: ข้าม (ยังไม่มีผล {src.name})", flush=True)
        return []
    # ตรวจอินพุตของทุกไฟล์ก่อนโหลดโมเดล (ภาพย่อยล้าสมัย/ไม่มี = หยุดโดยไม่เสียการโหลด GPU)
    for stem in stems:
        lay = json.loads((layout_dir / f"{stem}.json").read_text(encoding="utf-8"))["layout_sections"]
        if variant == "color":
            if load_final_image(paths, dataset, stem) is None:
                sys.exit(f"{stem[-4:]}: ไม่พบภาพ preprocess (final) — รัน fin_extractor preprocess ก่อน")
        else:
            clean_index(paths.section_images(dataset, stem, variant), lay, stem)
    if not redo and all(_is_done(doc_dir(raw_dir, s, nest=True) / f"{s}_p1.sections.json", layout_dir / f"{s}.json") for s in stems):
        print(f"ข้ามขั้น OCR: ทุกไฟล์ ({len(stems)}) มีผลที่ตรงกับผังปัจจุบันแล้ว ไม่โหลดโมเดล → {raw_dir}", flush=True)
        return []
    reader = make_reader(engine, cfg)
    print(f"{len(stems)} ไฟล์ → {raw_dir}", flush=True)
    t0 = time.perf_counter()
    if hasattr(reader, "_load"):  # typhoon เรียก Ollama ไม่มีโมเดลให้โหลดในเครื่อง
        reader._load()
    print(f"โหลดโมเดล {time.perf_counter() - t0:.1f}s · GPU {gpu_status()}", flush=True)

    summary = []
    t_start = time.perf_counter()
    torch = None
    if engine in TORCH_ENGINES:
        import torch

    for n, stem in enumerate(stems, start=1):
        out = doc_dir(raw_dir, stem, nest=True) / f"{stem}_p1.sections.json"
        done: dict[str, dict] = {}
        layout = json.loads((layout_dir / f"{stem}.json").read_text(encoding="utf-8"))["layout_sections"]
        if out.exists() and not redo:
            prev = json.loads(out.read_text(encoding="utf-8"))
            if prev.get("layout") != fingerprint(layout):  # ผังเปลี่ยน (หรือผลเก่าไม่มีลายนิ้วมือ) ใช้ต่อไม่ได้ ทำใหม่ทั้งไฟล์
                print(f"[{n}/{len(stems)}] {stem[-4:]} ผลเดิมไม่ตรงกับผังปัจจุบัน (หรือไม่มีบันทึกผัง) ทำใหม่ทั้งไฟล์", flush=True)
            elif prev.get("complete"):
                print(f"[{n}/{len(stems)}] {stem[-4:]} ข้าม (มีผลแล้ว ตรงกับผังปัจจุบัน)", flush=True)
                continue
            else:  # ค้างจากรอบก่อน (ผังเดิม): ต่อจากภาพที่ทำเสร็จแล้ว (ไม่นับภาพที่ error)
                done = {r["id"]: r for r in prev["sections"] if "error" not in r}
                print(f"[{n}/{len(stems)}] {stem[-4:]} ต่อจากที่ค้าง ({len(done)} ส่วนเสร็จแล้ว)", flush=True)
        if variant == "color":
            full = load_final_image(paths, dataset, stem)
            work, _ = resize_to_work(full)
        else:
            clean_dir = paths.section_images(dataset, stem, variant)
            entries = clean_index(clean_dir, layout, stem)
        rows, t_file = [], time.perf_counter()
        out.parent.mkdir(parents=True, exist_ok=True)

        def save(complete: bool) -> None:
            out.write_text(json.dumps({"source": stem, "page": 1, "engine": reader.name, "variant": variant,
                                       "max_new_tokens": getattr(reader, "max_new_tokens", None),
                                       "layout": fingerprint(layout), "complete": complete, "sections": rows}, ensure_ascii=False, indent=1), encoding="utf-8")

        for s in layout:
            if s["id"] in done:
                rows.append(done[s["id"]])
                continue
            row = {"id": s["id"], "kind": s["kind"], "bbox": s["bbox"], "table": s["table"],
                   **({"part": s["part"], "parts": s["parts"]} if s.get("part") else {})}
            if not s["keep"]:
                rows.append({**row, "skipped": "blank"})
                continue
            if variant != "color" and not entries[s["id"]].get("file"):  # เช่น bin ที่ทาดำหมดส่วน (--drop-empty-rows)
                rows.append({**row, "skipped": "blank"})
                continue
            img = crop_full(full, work, s) if variant == "color" else read_clean_image(clean_dir, entries[s["id"]], variant)
            t = time.perf_counter()
            try:
                res = reader.recognize(img)
                row.update(text=res.markdown, truncated=res.truncated)
            except Exception as e:  # noqa: BLE001
                row.update(text=None, error=repr(e)[:300])
                rows.append(row)
                save(False)
                # CUDA error แล้ว context มักเสีย รันต่อได้ผลเสียทั้งชุด: หยุด เก็บที่ทำไว้ แล้วรันใหม่จะต่อจากภาพนี้
                sys.exit(f"หยุด: {stem[-4:]} {s['id']} ({img.shape[1]}x{img.shape[0]}) {type(e).__name__}: {str(e)[:160]}\n"
                         f"VRAM ตอนนั้น: {gpu_status()} — รันคำสั่งเดิมซ้ำเพื่อต่อจากภาพนี้ (ภาพก่อนหน้าเก็บไว้แล้ว)")
            row["seconds"] = round(time.perf_counter() - t, 2)
            rows.append(row)
            save(False)  # บันทึกทีละภาพ ล้มกลางทางไม่เสียงานที่ทำไป
            reserved = ""
            if torch is not None:
                torch.cuda.empty_cache()  # ปล่อยหน่วยความจำที่ PyTorch เก็บสะสมไว้ ไม่ให้ cuDNN ขอเพิ่มไม่ได้ในภาพถัดไป
                reserved = f"reserved {torch.cuda.memory_reserved() / 2**30:.2f} GiB · "
            print(f"    {stem[-4:]} {s['id']} {s['kind']:4s} {img.shape[1]}x{img.shape[0]} {row['seconds']:5.1f}s "
                  f"{'ชนเพดาน token ' if row['truncated'] else ''}| {reserved}GPU {gpu_status()}", flush=True)
        save(True)
        write_markdown(out, image_locator(paths, dataset, stem, variant))
        ran = [r for r in rows if "skipped" not in r]
        info = {"file": stem[-4:], "sections": len(rows), "ran": len(ran), "skipped": len(rows) - len(ran),
                "truncated": sum(bool(r.get("truncated")) for r in ran), "seconds": round(time.perf_counter() - t_file, 1)}
        summary.append(info)
        print(f"[{n}/{len(stems)}] {info['file']} ส่วน {info['sections']} (อ่าน {info['ran']}, ข้าม {info['skipped']}) "
              f"ชนเพดาน token {info['truncated']} · {info['seconds']}s · GPU {gpu_status()} "
              f"· รวม {(time.perf_counter() - t_start) / 60:.1f} นาที", flush=True)
    return summary
