"""Command line: ``python -m fin_extractor <command> ...``

  preprocess  ปรับภาพแล้วเก็บภาพสุดท้ายไว้ให้ทุก engine ใช้ร่วมกัน (ดูผลทีละขั้นด้วย --steps)
  ocr         อ่านข้อความดิบด้วย OCR engine หรือ VLM แล้วเก็บเป็น raw_data
  extract     pipeline เต็ม → JSON ตาม schema (--reuse-raw = สกัดซ้ำจากข้อความดิบที่เก็บไว้ ไม่เรียก OCR)
  schema      export JSON Schema

ผลอยู่ใต้ data/ ตามโครงใน fin_extractor/paths.py:
  preprocess/<dataset>/<profile>/{final,steps}/    result_ocr/<engine>/<dataset>/{raw_data,final/<extractor>}/
(รันที่มีไฟล์อินพุตเกิน 2 ไฟล์ ผลแต่ละไฟล์อยู่ในโฟลเดอร์ย่อยชื่อไฟล์ ดู paths.nest_per_file)
พร้อม manifest.json (รายละเอียดดู fin_extractor/runs.py)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from .config import load_config, resolve_path
from .io import iter_input_files
from .ocr import ENGINES
from .paths import DEFAULT_PROFILE, DataPaths, dataset_of, doc_dir, find_dir, nest_per_file
from .runs import Run
from .stages import load_raw, prepared_pages, raw_pages, save_raw


def engine_label(mode: str, engine: str) -> str:
    """ชื่อโฟลเดอร์ engine ใน result_ocr/ (mode vlm = typhoon)."""
    return "typhoon" if mode == "vlm" else engine


def cmd_preprocess(args, cfg) -> None:
    paths, dataset = DataPaths(cfg), dataset_of(args.input)
    pdir = Path(args.out) if args.out else paths.preprocess(dataset, args.profile)
    with Run("preprocess", args.input, cfg, pdir, {"profile": args.profile, "steps": args.steps, "force": args.force}) as run:
        files = list(iter_input_files(args.input))
        for f in files:
            for page_no, _img, meta in prepared_pages(f, cfg, pdir, steps=args.steps, force=args.force, nest=nest_per_file(len(files))):
                out = find_dir(pdir / "final", f.stem) / f"{f.stem}_p{page_no}.png"
                run.add(file=f.name, page=page_no, output=out.relative_to(run.dir).as_posix(), meta=meta)
                print(f"{out}  {meta}")
    print(f"manifest: {run.dir / 'manifest.json'}")


def cmd_ocr(args, cfg) -> None:
    paths, dataset = DataPaths(cfg), dataset_of(args.input)
    if args.engine == "vlm":
        from .vlm import TyphoonOCR

        reader = TyphoonOCR(cfg["vlm"])
    else:
        from .ocr import create_engine

        reader = create_engine(args.engine, cfg)

    mode = "vlm" if args.engine == "vlm" else "ocr"
    label = engine_label(mode, args.engine)
    ext = "md" if mode == "vlm" else "txt"
    raw_dir = Path(args.out) / "raw_data" if args.out else paths.raw_data(label, dataset)
    pdir = paths.preprocess(dataset, args.profile)
    settings = {"engine": label, "profile": args.profile, **({"model": cfg["vlm"]["model"]} if mode == "vlm" else {})}
    with Run("ocr", args.input, cfg, raw_dir, settings) as run:
        files = list(iter_input_files(args.input))
        nest = nest_per_file(len(files))
        for f in files:
            for page_no, img, _ in prepared_pages(f, cfg, pdir, nest=nest):
                t0 = time.perf_counter()
                res = reader.recognize(img)
                text = res.markdown if mode == "vlm" else res.text
                conf = getattr(res, "mean_confidence", None)
                seconds = round(time.perf_counter() - t0, 2)
                meta = {"engine": label, "ocr_confidence": round(conf, 4) if conf is not None else None,
                        "truncated": bool(getattr(res, "truncated", False)), "profile": args.profile, "seconds": seconds}
                out = save_raw(doc_dir(raw_dir, f.stem, nest), f"{f.stem}_p{page_no}", text, ext, meta)
                run.add(file=f.name, page=page_no, output=out.relative_to(run.dir).as_posix(),
                        mean_confidence=meta["ocr_confidence"], seconds=seconds)
                print(f"--- {f.name} p{page_no} [{label}] {seconds:.1f}s"
                      + (f" conf={conf:.3f}" if conf is not None else "") + f" → {out}")
                print(text)
    print(f"manifest: {run.dir / 'manifest.json'}")


def cmd_extract(args, cfg) -> None:
    from .pipeline import FinanceOCRPipeline

    paths, dataset = DataPaths(cfg), dataset_of(args.input)
    use_llm = {"on": True, "off": False, "auto": None}[args.llm]
    pipe = FinanceOCRPipeline(cfg, mode=args.mode, engine=args.engine, use_llm=use_llm, with_reader=not args.reuse_raw)
    label, extractor = engine_label(args.mode, args.engine), "llm" if pipe.llm else "rules"
    ext = "md" if args.mode == "vlm" else "txt"
    base = Path(args.out) if args.out else None
    raw_dir = base / "raw_data" if base else paths.raw_data(label, dataset)
    final_dir = base / "final" / extractor if base else paths.final(label, dataset, extractor)
    pdir = paths.preprocess(dataset, args.profile)
    settings = {
        "mode": args.mode, "engine": pipe.engine_name, "profile": args.profile, "reuse_raw": args.reuse_raw,
        "keep_raw": args.keep_raw, "raw_data": Path(os.path.relpath(raw_dir, final_dir)).as_posix(),
        **({"vlm_model": cfg["vlm"]["model"]} if args.mode == "vlm" else {}),
        **({"llm_model": cfg["llm_extractor"]["model"]} if pipe.llm else {}),
    }
    with Run("extract", args.input, cfg, final_dir, settings) as run:
        files = list(iter_input_files(args.input))
        nest = nest_per_file(len(files))
        for f in files:
            t0 = time.perf_counter()
            if args.reuse_raw:
                pages = raw_pages(raw_dir, f.stem, ext)
                if not pages:
                    print(f"ข้าม {f.name}: ไม่มีข้อความดิบใน {raw_dir} (รัน extract โดยไม่ใส่ --reuse-raw ก่อน)")
                    continue
                results = ((no, pipe.process_raw(str(f), no, *load_raw(p))) for no, p in pages)
            else:
                results = ((no, pipe.process_page(str(f), no, img)) for no, img, _ in prepared_pages(f, cfg, pdir, nest=nest))
            for page_no, r in results:
                seconds = round(time.perf_counter() - t0, 2)
                if not args.reuse_raw:  # ข้อความดิบเก็บแยกที่ raw_data (OCR ครั้งเดียว สกัดได้หลายแบบ)
                    meta = {"engine": r.engine, "ocr_confidence": r.ocr_confidence, "profile": args.profile, "seconds": seconds,
                            "truncated": any(c.name == "vlm_complete" and c.passed is False for c in r.checks)}
                    save_raw(doc_dir(raw_dir, f.stem, nest), f"{f.stem}_p{page_no}", r.raw_text or "", ext, meta)
                if not args.keep_raw:
                    r.raw_text = None
                out = doc_dir(final_dir, f.stem, nest) / f"{f.stem}_p{r.page}.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(r.model_dump_json(indent=2), encoding="utf-8")
                passed = sum(c.passed is True for c in r.checks)
                failed = sum(c.passed is False for c in r.checks)
                run.add(file=f.name, page=r.page, status=r.status, missing_fields=r.missing_fields,
                        checks_passed=passed, checks_failed=failed, seconds=seconds,
                        output=out.relative_to(run.dir).as_posix())
                print(f"{f.name} p{r.page}: {r.status} checks={passed}✓ {failed}✗ "
                      f"missing={','.join(r.missing_fields) or '-'} ({seconds:.1f}s) → {out}")
                for c in r.checks:
                    mark = {True: "✓", False: "✗", None: "·"}[c.passed]
                    print(f"   {mark} {c.name}: {c.detail}")
                t0 = time.perf_counter()
    print(f"manifest: {run.dir / 'manifest.json'}")


def cmd_schema(args, cfg) -> None:
    from .schema import ExtractionResult

    out = Path(args.out or resolve_path("schemas/invoice.schema.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ExtractionResult.model_json_schema(), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fin_extractor", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="YAML ที่ override configs/default.yaml")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    def profile_arg(s) -> None:
        s.add_argument("--profile", default=DEFAULT_PROFILE,
                       help="ชื่อชุดตั้งค่า preprocess (โฟลเดอร์ preprocess/<dataset>/<profile>/) ค่าตั้งต้น default")

    s = sub.add_parser("preprocess", help="ปรับภาพก่อน OCR (เก็บภาพสุดท้ายใน final/)")
    s.add_argument("input", help="ไฟล์หรือโฟลเดอร์")
    profile_arg(s)
    s.add_argument("--steps", action="store_true", help="บันทึกภาพทุกขั้นตอนใน steps/ (ทำใหม่แม้มีผลเก็บไว้)")
    s.add_argument("--force", action="store_true", help="ทำใหม่ทับ ถ้าตั้งค่าเปลี่ยนจากที่เก็บไว้ = ล้างโฟลเดอร์ profile นั้นก่อน")
    s.add_argument("--out", help="โฟลเดอร์ profile เอง (ค่าตั้งต้น data/preprocess/<dataset>/<profile>)")
    s.set_defaults(func=cmd_preprocess)

    s = sub.add_parser("ocr", help="อ่านข้อความดิบ → raw_data")
    s.add_argument("input")
    s.add_argument("--engine", choices=[*ENGINES, "vlm"], default="paddle")
    profile_arg(s)
    s.add_argument("--out", help="โฟลเดอร์ engine เอง (ค่าตั้งต้น data/result_ocr/<engine>/<dataset>) ข้อความดิบอยู่ใต้ raw_data/")
    s.set_defaults(func=cmd_ocr)

    s = sub.add_parser("extract", help="สกัด field → JSON")
    s.add_argument("input")
    s.add_argument("--mode", choices=["ocr", "vlm"], default="ocr")
    s.add_argument("--engine", choices=ENGINES, default="paddle", help="ใช้เมื่อ --mode ocr (ค่าตั้งต้น paddle)")
    s.add_argument("--llm", choices=["auto", "on", "off"], default="auto",
                   help="auto = ใช้ LLM เฉพาะ mode vlm")
    profile_arg(s)
    s.add_argument("--reuse-raw", action="store_true",
                   help="สกัดซ้ำจากข้อความดิบที่เก็บไว้ใน raw_data (ไม่โหลด/เรียก OCR หรือ VLM ใช้ลองเปลี่ยน rules/LLM)")
    s.add_argument("--keep-raw", action="store_true", help="เก็บข้อความดิบใน JSON ผลสุดท้ายด้วย")
    s.add_argument("--out", help="โฟลเดอร์ engine เอง (ค่าตั้งต้น data/result_ocr/<engine>/<dataset>) ผลอยู่ใต้ raw_data/ และ final/<extractor>/")
    s.set_defaults(func=cmd_extract)

    s = sub.add_parser("schema", help="export JSON Schema")
    s.add_argument("--out")
    s.set_defaults(func=cmd_schema)
    return p


def main(argv: list[str] | None = None) -> None:
    # Windows console default เป็น cp874/cp1252 → print ภาษาไทยแล้วพัง
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    args.func(args, load_config(args.config))


if __name__ == "__main__":
    main()
