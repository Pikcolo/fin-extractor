"""รัน OCR ทีละภาพย่อย (section) ของทุกไฟล์ที่แบ่งไว้ด้วย notebooks/preprocess_lab แล้วเก็บเป็น raw_data

    python -X utf8 scripts/run_sections.py data/raw/finance_doc_scan                     # --engine openthai --variant luma (ค่าตั้งต้น)
    python -X utf8 scripts/run_sections.py data/raw/finance_doc_scan --engine oxr
    python -X utf8 scripts/run_sections.py data/raw/finance_doc_scan --variant flat --only 1307     # อ่านภาพ sections/clean/flat/
    python -X utf8 scripts/run_sections.py data/raw/finance_doc_scan --variant color               # ตัดจากภาพสีใน memory (ไม่ผ่าน luma/flat)
    python -X utf8 scripts/run_sections.py data/raw/finance_doc_scan --only 1309 1311 --redo

อินพุต: ผังแบ่ง ``sections/features/<ไฟล์>.json`` (make_section_layouts.py) + ภาพ ``--variant``
    luma (ค่าตั้งต้น) / flat / bin (ตัวขาวบนดำ) = ไฟล์ภาพย่อย ``sections/clean/<ไฟล์>/sNN_<แบบ>.png`` (make_clean_sections.py) ตรวจก่อนว่าตรงกับผังปัจจุบัน ไม่งั้นหยุด
    color                    = ตัดจากภาพ final สีในหน่วยความจำตามผัง ไม่มีไฟล์ภาพย่อย ไม่ผ่าน luma/flat
ตัวอ่าน (เลือกด้วย ``--engine``) ตั้งค่าที่ section ชื่อเดียวกันใน configs/default.yaml
    oxr      ``vlm/oxr_local.py``    (``oxr_local:``) โหลดตรง ไม่ใช้ Docker
    openthai ``vlm/openthai_ocr.py`` (``openthai:``)
ผล: ``data/result_ocr/<engine>/<dataset>/raw_data[_<แบบ>]/<ไฟล์>/<ไฟล์>_p1.sections.json`` = รายการ section ตามลำดับอ่าน (color = raw_data)
    {id, kind, bbox, text, truncated, seconds} ส่วนที่ lab ตัดทิ้ง (ว่าง) ไม่ถูกอ่าน เก็บเป็น ``skipped``
นอกจากนี้เขียน ``<ไฟล์>_p1.md`` ไว้ข้างกัน = ข้อความของทุกส่วนต่อกันตามลำดับอ่าน (ป้ายแต่ละส่วนเป็นคอมเมนต์ HTML ซึ่ง rules ตัดทิ้งตอนอ่าน)
เป็นรูปแบบเดียวกับข้อความดิบของโหมด VLM จึงใช้กับ ``extract --reuse-raw`` ได้ ``--markdown-only`` สร้าง .md จากผลที่มีอยู่โดยไม่โหลดโมเดล (ไม่ใช้ GPU)
รันต่อจากที่ค้างได้ (ข้ามไฟล์ที่มีผลแล้ว ใส่ --redo เพื่อทำใหม่) บันทึกผลทีละไฟล์

OXR ไม่รู้ภาษาไทย: ช่องที่มีไทยจะได้ข้อความมั่ว/อักษรจีน และอาจวนซ้ำจนชนเพดาน token (ช้า) ผลนี้เป็นข้อมูลดิบ ยังไม่ได้คัดว่าช่องไหนเชื่อได้
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fin_extractor.config import load_config  # noqa: E402
from fin_extractor.paths import DataPaths, dataset_of, doc_dir  # noqa: E402

sys.path.insert(0, str(ROOT / "notebooks" / "preprocess_lab"))
import labkit as L  # noqa: E402  (ตัดภาพย่อยจากภาพทั้งหน้าตามผัง ไม่มีไฟล์ภาพย่อยแล้ว)

ENGINES = ("oxr", "openthai")
VARIANTS = ("color", "luma", "flat", "bin")


def kept_layout(layout: list[dict]) -> list[tuple[str, list[int]]]:
    return [(s["id"], list(s["bbox"])) for s in layout if s["keep"]]


def fingerprint(layout: list[dict]) -> list[list]:
    """ส่วนที่เก็บของผังปัจจุบัน (id + bbox) บันทึกไว้ในผลทุกไฟล์ ใช้ตัดสินว่าผลเก่าข้ามหรือต่อได้ไหม — ผังเปลี่ยนแล้วผลเก่าไม่ตรงอีกต่อไป."""
    return [[i, b] for i, b in kept_layout(layout)]


def clean_index(clean_dir: Path, layout: list[dict], stem: str) -> dict[str, dict]:
    """index.json ของภาพย่อยใน sections/clean/ (id → รายการ) หยุดถ้าไม่ตรงกับผังปัจจุบัน — ภาพล้าสมัยจะอ่านผิดส่วนโดยไม่รู้ตัว."""
    path = clean_dir / "index.json"
    if not path.exists():
        sys.exit(f"{stem[-4:]}: ไม่มีภาพย่อยใน {clean_dir} — รัน scripts/make_clean_sections.py ก่อน")
    index = json.loads(path.read_text(encoding="utf-8"))["sections"]
    if [(s["id"], list(s["bbox"])) for s in index] != kept_layout(layout):
        sys.exit(f"{stem[-4:]}: ภาพย่อยใน {clean_dir} ไม่ตรงกับผังแบ่งปัจจุบัน (สร้างจากผังเก่า) — รัน scripts/make_clean_sections.py ใหม่")
    return {s["id"]: s for s in index}


def read_clean_image(clean_dir: Path, entry: dict, variant: str) -> np.ndarray:
    """ภาพเทาช่องเดียวจากไฟล์ → BGR (ตัวอ่านรับภาพ BGR เหมือน route_sections.py)."""
    gray = cv2.imdecode(np.fromfile(str(clean_dir / entry["file"]), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def make_reader(engine: str, cfg: dict):
    """ตัวอ่านของ engine นั้น (import ตอนใช้ ไม่โหลด torch/โมเดลของตัวที่ไม่ได้เลือก)."""
    if engine == "oxr":
        from fin_extractor.vlm.oxr_local import OXRLocal
        return OXRLocal(cfg["oxr_local"])
    from fin_extractor.vlm.openthai_ocr import OpenThaiOCR
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


def remark_of(doc: dict, image_of) -> dict:
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
    """เขียน .md (ข้อความดิบที่ extract --reuse-raw อ่าน) และ .remark.json (ที่มาของแต่ละท่อน ไม่ถูกอ่านโดย extract)."""
    stem = sections_json.name.removesuffix(".sections.json")
    doc = json.loads(sections_json.read_text(encoding="utf-8"))
    out = sections_json.with_name(stem + ".md")
    out.write_text(markdown_of(doc), encoding="utf-8")
    sections_json.with_name(stem + ".remark.json").write_text(json.dumps(remark_of(doc, image_of), ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def image_locator(paths: DataPaths, dataset: str, stem: str, variant: str):
    """ฟังก์ชันที่บอกที่อยู่ภาพของ section (สัมพัทธ์กับ data_dir) ใช้ใน .remark.json."""
    if variant == "color":
        return lambda s: None
    entries = {e["id"]: e for e in json.loads((paths.section_clean(dataset, stem, variant) / "index.json").read_text(encoding="utf-8"))["sections"]}
    base = paths.section_clean(dataset, stem, variant)
    data_dir = paths.preprocess_final(dataset).parents[3]  # data/preprocess/<ชุด>/<โปรไฟล์>/final → data/

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="โฟลเดอร์ชุดข้อมูลเดียวกับที่ใช้สร้าง notebook (ใช้เอาชื่อ dataset)")
    ap.add_argument("--only", nargs="*", help="เลขท้ายชื่อไฟล์ เช่น 1309 (ไม่ใส่ = ทุกไฟล์ที่มีผังแบ่ง)")
    ap.add_argument("--redo", action="store_true", help="ทำใหม่แม้มีผลแล้ว")
    ap.add_argument("--markdown-only", action="store_true", help="สร้าง .md จากผลที่มีอยู่ ไม่โหลดโมเดล ไม่อ่านใหม่ (ไม่ใช้ GPU)")
    ap.add_argument("--config", help="ไฟล์ config ทับ default.yaml (เปลี่ยน paths.data_dir ได้ ใช้ทดลองโดยไม่ทับผลจริง)")
    ap.add_argument("--engine", choices=ENGINES, default="openthai", help="ตัวอ่านที่ใช้ (ผลลงใต้ result_ocr/<engine>/)")
    ap.add_argument("--variant", choices=VARIANTS, default="luma",
                    help="ภาพที่ส่งเข้าตัวอ่าน: luma (ค่าตั้งต้น) / flat / bin (ตัวขาวบนดำ) = ไฟล์ใน sections/clean/ (สร้างด้วย make_clean_sections.py) · "
                         "color = ตัดจากภาพ final สีในหน่วยความจำ (ไม่ผ่านการเตรียมรายส่วน) ผลแต่ละแบบแยกโฟลเดอร์ ไม่ทับกัน")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = DataPaths(cfg)
    L.PATHS = paths  # labkit (ตัดภาพ final ตามผัง) ถามพาธจากตัวแปรนี้
    dataset = dataset_of(args.input)
    features_dir = paths.section_features(dataset)
    stems = sorted(p.stem for p in features_dir.glob("*.json"))
    if args.only:
        stems = [s for s in stems if any(s.endswith(o) for o in args.only)]
    if not stems:
        sys.exit(f"ไม่พบผังแบ่งใน {features_dir} — รัน notebook/lab ก่อน")

    # color = ที่เดิม (raw_data) · luma/flat = raw_data_<variant> ข้างกัน: เทียบกันได้โดยไม่ทับ และ "มีผลแล้วให้ข้าม" นับแยกตามแบบ
    raw_dir = paths.raw_data(args.engine, dataset)
    if args.variant != "color":
        raw_dir = raw_dir.with_name(f"{raw_dir.name}_{args.variant}")
    if args.markdown_only:
        for stem in stems:
            src = doc_dir(raw_dir, stem, nest=True) / f"{stem}_p1.sections.json"
            if src.exists():
                print(f"{stem[-4:]}: {write_markdown(src, image_locator(paths, dataset, stem, args.variant))}", flush=True)
            else:
                print(f"{stem[-4:]}: ข้าม (ยังไม่มีผล {src.name})", flush=True)
        return
    # ตรวจอินพุตของทุกไฟล์ก่อนโหลดโมเดล (ภาพย่อยล้าสมัย/ไม่มี = หยุดโดยไม่เสียการโหลด GPU)
    for stem in stems:
        lay = json.loads((features_dir / f"{stem}.json").read_text(encoding="utf-8"))["layout_sections"]
        if args.variant == "color":
            if L.load_cached_preprocessed(stem, dataset=dataset) is None:
                sys.exit(f"{stem[-4:]}: ไม่พบภาพ preprocess (final) — รัน fin_extractor preprocess ก่อน")
        else:
            clean_index(paths.section_clean(dataset, stem, args.variant), lay, stem)
    reader = make_reader(args.engine, cfg)
    print(f"{len(stems)} ไฟล์ → {raw_dir}", flush=True)
    t0 = time.perf_counter()
    reader._load()
    print(f"โหลดโมเดล {time.perf_counter() - t0:.1f}s · GPU {gpu_status()}", flush=True)

    summary = []
    t_start = time.perf_counter()
    import torch

    for n, stem in enumerate(stems, start=1):
        out = doc_dir(raw_dir, stem, nest=True) / f"{stem}_p1.sections.json"
        done: dict[str, dict] = {}
        layout = json.loads((features_dir / f"{stem}.json").read_text(encoding="utf-8"))["layout_sections"]
        if out.exists() and not args.redo:
            prev = json.loads(out.read_text(encoding="utf-8"))
            if prev.get("layout") != fingerprint(layout):  # ผังเปลี่ยน (หรือผลเก่าไม่มีลายนิ้วมือ) ใช้ต่อไม่ได้ ทำใหม่ทั้งไฟล์
                print(f"[{n}/{len(stems)}] {stem[-4:]} ผลเดิมไม่ตรงกับผังปัจจุบัน (หรือไม่มีบันทึกผัง) ทำใหม่ทั้งไฟล์", flush=True)
            elif prev.get("complete"):
                print(f"[{n}/{len(stems)}] {stem[-4:]} ข้าม (มีผลแล้ว ตรงกับผังปัจจุบัน)", flush=True)
                continue
            else:  # ค้างจากรอบก่อน (ผังเดิม): ต่อจากภาพที่ทำเสร็จแล้ว (ไม่นับภาพที่ error)
                done = {r["id"]: r for r in prev["sections"] if "error" not in r}
                print(f"[{n}/{len(stems)}] {stem[-4:]} ต่อจากที่ค้าง ({len(done)} ส่วนเสร็จแล้ว)", flush=True)
        if args.variant == "color":
            full = L.load_cached_preprocessed(stem, dataset=dataset)
            work, _ = L._resize_to_work(full)
        else:
            clean_dir = paths.section_clean(dataset, stem, args.variant)
            entries = clean_index(clean_dir, layout, stem)
        rows, t_file = [], time.perf_counter()
        out.parent.mkdir(parents=True, exist_ok=True)

        def save(complete: bool) -> None:
            out.write_text(json.dumps({"source": stem, "page": 1, "engine": reader.name, "variant": args.variant,
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
            if args.variant != "color" and not entries[s["id"]].get("file"):  # เช่น bin ที่ทาดำหมดส่วน (--drop-empty-rows)
                rows.append({**row, "skipped": "blank"})
                continue
            img = L.crop_full(full, work, s) if args.variant == "color" else read_clean_image(clean_dir, entries[s["id"]], args.variant)
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
            torch.cuda.empty_cache()  # ปล่อยหน่วยความจำที่ PyTorch เก็บสะสมไว้ ไม่ให้ cuDNN ขอเพิ่มไม่ได้ในภาพถัดไป
            print(f"    {stem[-4:]} {s['id']} {s['kind']:4s} {img.shape[1]}x{img.shape[0]} {row['seconds']:5.1f}s "
                  f"{'ชนเพดาน token ' if row['truncated'] else ''}| reserved {torch.cuda.memory_reserved() / 2**30:.2f} GiB · GPU {gpu_status()}", flush=True)
        save(True)
        write_markdown(out, image_locator(paths, dataset, stem, args.variant))
        ran = [r for r in rows if "skipped" not in r]
        info = {"file": stem[-4:], "sections": len(rows), "ran": len(ran), "skipped": len(rows) - len(ran),
                "truncated": sum(bool(r.get("truncated")) for r in ran), "seconds": round(time.perf_counter() - t_file, 1)}
        summary.append(info)
        print(f"[{n}/{len(stems)}] {info['file']} ส่วน {info['sections']} (อ่าน {info['ran']}, ข้าม {info['skipped']}) "
              f"ชนเพดาน token {info['truncated']} · {info['seconds']}s · GPU {gpu_status()} "
              f"· รวม {(time.perf_counter() - t_start) / 60:.1f} นาที", flush=True)

    print("\nสรุป:")
    for i in summary:
        print(i)
    print(f"ไฟล์ที่รันรอบนี้ {len(summary)} · เวลารวม {(time.perf_counter() - t_start) / 60:.1f} นาที")


if __name__ == "__main__":
    main()
