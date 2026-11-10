"""Command line: ``python -m fin_extractor <command> ...``

  preprocess  ปรับภาพแล้วบันทึก (ดูผลแต่ละขั้นด้วย --steps)
  ocr         อ่านข้อความดิบด้วย OCR engine หรือ VLM
  extract     pipeline เต็ม → JSON ตาม schema
  schema      export JSON Schema

ผลของ preprocess/ocr/extract อยู่ที่ data/output/<คำสั่ง>/<ชื่ออินพุต>__<แท็ก>/ พร้อม manifest.json
(โครงละเอียดดู fin_extractor/runs.py)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from .config import load_config, resolve_path
from .io import iter_input_files, load_pages, save_image
from .ocr import ENGINES
from .runs import Run


def cmd_preprocess(args, cfg) -> None:
    from .preprocessing import preprocess

    with Run("preprocess", args.input, None, cfg, args.out, {"steps": args.steps}) as run:
        images = run.sub("images")
        for f in iter_input_files(args.input):
            for page_no, img in enumerate(load_pages(f, cfg["loader"]["pdf_dpi"]), start=1):
                stem = f"{f.stem}_p{page_no}"
                res = preprocess(img, cfg["preprocessing"], keep_steps=args.steps)
                out = images / f"{stem}.png"
                save_image(out, res.image)
                if args.steps:
                    res.save_steps(run.sub("steps"), stem)
                run.add(file=f.name, page=page_no, output=out.relative_to(run.dir).as_posix(), meta=res.meta)
                print(f"{out}  {res.meta}")
    print(f"manifest: {run.dir / 'manifest.json'}")


def cmd_ocr(args, cfg) -> None:
    from .preprocessing import preprocess

    if args.engine == "vlm":
        from .vlm import TyphoonOCR

        reader = TyphoonOCR(cfg["vlm"])
    else:
        from .ocr import create_engine

        reader = create_engine(args.engine, cfg)

    ext = "md" if args.engine == "vlm" else "txt"
    settings = {"engine": args.engine, **({"model": cfg["vlm"]["model"]} if args.engine == "vlm" else {})}
    with Run("ocr", args.input, args.engine, cfg, args.out, settings) as run:
        texts = run.sub("text")
        for f in iter_input_files(args.input):
            for page_no, img in enumerate(load_pages(f, cfg["loader"]["pdf_dpi"]), start=1):
                t0 = time.perf_counter()
                res = reader.recognize(preprocess(img, cfg["preprocessing"]).image)
                text = res.markdown if args.engine == "vlm" else res.text
                conf = getattr(res, "mean_confidence", None)
                out = texts / f"{f.stem}_p{page_no}.{ext}"
                out.write_text(text, encoding="utf-8")
                seconds = round(time.perf_counter() - t0, 2)
                run.add(file=f.name, page=page_no, output=out.relative_to(run.dir).as_posix(),
                        mean_confidence=round(conf, 4) if conf is not None else None, seconds=seconds)
                print(f"--- {f.name} p{page_no} [{args.engine}] {seconds:.1f}s"
                      + (f" conf={conf:.3f}" if conf is not None else "") + f" → {out}")
                print(text)
    print(f"manifest: {run.dir / 'manifest.json'}")


def cmd_extract(args, cfg) -> None:
    from .pipeline import FinanceOCRPipeline

    use_llm = {"on": True, "off": False, "auto": None}[args.llm]
    pipe = FinanceOCRPipeline(cfg, mode=args.mode, engine=args.engine, use_llm=use_llm)
    settings = {
        "mode": args.mode, "engine": pipe.engine_name, "keep_raw": args.keep_raw,
        **({"vlm_model": cfg["vlm"]["model"]} if args.mode == "vlm" else {}),
        **({"llm_model": cfg["llm_extractor"]["model"]} if pipe.llm else {}),
    }
    with Run("extract", args.input, f"{args.mode}-{'llm' if pipe.llm else 'rules'}", cfg, args.out, settings) as run:
        results = run.sub("results")
        debug_dir = run.dir / "debug" if args.debug else None
        for f in iter_input_files(args.input):
            t0 = time.perf_counter()
            for r in pipe.process_file(f, debug_dir):
                if not args.keep_raw:
                    r.raw_text = None
                out = results / f"{f.stem}_p{r.page}.json"
                out.write_text(r.model_dump_json(indent=2), encoding="utf-8")
                passed = sum(c.passed is True for c in r.checks)
                failed = sum(c.passed is False for c in r.checks)
                seconds = round(time.perf_counter() - t0, 2)
                run.add(file=f.name, page=r.page, status=r.status, missing_fields=r.missing_fields,
                        checks_passed=passed, checks_failed=failed, seconds=seconds,
                        output=out.relative_to(run.dir).as_posix())
                print(f"{f.name} p{r.page}: {r.status} checks={passed}✓ {failed}✗ "
                      f"missing={','.join(r.missing_fields) or '-'} ({seconds:.1f}s) → {out}")
                for c in r.checks:
                    mark = {True: "✓", False: "✗", None: "·"}[c.passed]
                    print(f"   {mark} {c.name}: {c.detail}")
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

    s = sub.add_parser("preprocess", help="ปรับภาพก่อน OCR")
    s.add_argument("input", help="ไฟล์หรือโฟลเดอร์")
    s.add_argument("--steps", action="store_true", help="บันทึกภาพทุกขั้นตอน")
    s.add_argument("--out", help="โฟลเดอร์ผลทั้งโฟลเดอร์ (ค่าตั้งต้น data/output/preprocess/<ชื่ออินพุต>)")
    s.set_defaults(func=cmd_preprocess)

    s = sub.add_parser("ocr", help="อ่านข้อความดิบ")
    s.add_argument("input")
    s.add_argument("--engine", choices=[*ENGINES, "vlm"], default="paddle")
    s.add_argument("--out", help="โฟลเดอร์ผลทั้งโฟลเดอร์ (ค่าตั้งต้น data/output/ocr/<ชื่ออินพุต>__<engine>)")
    s.set_defaults(func=cmd_ocr)

    s = sub.add_parser("extract", help="สกัด field → JSON")
    s.add_argument("input")
    s.add_argument("--mode", choices=["ocr", "vlm"], default="ocr")
    s.add_argument("--engine", choices=ENGINES, default="paddle", help="ใช้เมื่อ --mode ocr (ค่าตั้งต้น paddle)")
    s.add_argument("--llm", choices=["auto", "on", "off"], default="auto",
                   help="auto = ใช้ LLM เฉพาะ mode vlm")
    s.add_argument("--debug", action="store_true", help="บันทึกภาพ preprocess ทุกขั้น")
    s.add_argument("--keep-raw", action="store_true", help="เก็บข้อความดิบใน JSON")
    s.add_argument("--out", help="โฟลเดอร์ผลทั้งโฟลเดอร์ (ค่าตั้งต้น data/output/extract/<ชื่ออินพุต>__<mode>-<llm|rules>)")
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
