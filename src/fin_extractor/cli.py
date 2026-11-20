"""Command line: ``python -m fin_extractor <command> ...``

  preprocess  ปรับภาพแล้วเก็บภาพสุดท้ายไว้ให้ทุก engine ใช้ร่วมกัน (ดูผลทีละขั้นด้วย --steps)
  sections    flow หลัก: preprocess → แบ่ง section → ภาพรายส่วน → อ่านทีละส่วน → สกัด (ทุก engine อ่านทีละ section เท่านั้น)
  extract     สกัดซ้ำจากข้อความรายส่วนที่เก็บไว้ (ไม่อ่านภาพ) · ocr ปิดแล้ว (ห้ามอ่านทั้งหน้า)
  compare     เทียบผลสกัดข้ามวิธี luma/flat/bin
  schema      export JSON Schema

ผลอยู่ใต้ data/ ตามโครงใน fin_extractor/paths.py:
  02_preprocess/<dataset>/<profile>/{final,steps}/   04_ocr/<dataset>/<engine>/<วิธี>/   05_extract/<dataset>/<engine>/<วิธี>/<extractor>/
  (วิธี = luma|flat|bin|color ภาพรายส่วนที่ส่งเข้า OCR · โฟลเดอร์ page/ ของรุ่นเก่าที่อ่านทั้งหน้าไม่ถูกสร้างอีก) ดู docs/03-sections.md
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
from .paths import DEFAULT_PROFILE, FINAL_DIR, DataPaths, dataset_of, doc_dir, find_dir, nest_per_file, variant_of
from .runs import Run
from .sections.read import ENGINES as SECTION_ENGINES
from .stages import load_raw, prepared_pages, raw_pages


def cmd_preprocess(args, cfg) -> None:
    paths, dataset = DataPaths(cfg), dataset_of(args.input)
    pdir = Path(args.out) if args.out else paths.preprocess(dataset, args.profile)
    with Run("preprocess", args.input, cfg, pdir, {"profile": args.profile, "steps": args.steps, "force": args.force}) as run:
        files = list(iter_input_files(args.input))
        for f in files:
            for page_no, _img, meta in prepared_pages(f, cfg, pdir, steps=args.steps, force=args.force, nest=nest_per_file(len(files)),
                                                      orientation=args.orientation):
                out = find_dir(pdir / FINAL_DIR, f.stem) / f"{f.stem}_p{page_no}.png"
                run.add(file=f.name, page=page_no, output=out.relative_to(run.dir).as_posix(), meta=meta)
                print(f"{out}  {meta}")
                if meta.get("orientation_error"):
                    print(f"คำเตือน: {f.name} หน้า {page_no} ตรวจทิศไม่ได้ ({meta['orientation_error'][:90]}) จึงใช้ 0° ภาพอาจตะแคงอยู่ "
                          "ตรวจภาพ final แล้วสั่งทิศเองได้ด้วย --orientation 90|180|270 --force (ไฟล์เดียว)")
    print(f"manifest: {run.dir / 'manifest.json'}")


WHOLE_PAGE_BLOCKED = ("ห้ามอ่านทั้งหน้า: ทุก engine อ่านทีละ section เท่านั้น — ใช้ "
                      "python -m fin_extractor sections <input> --engine {engine} (เฉพาะอ่าน: --to ocr · สกัดซ้ำจากข้อความที่มี: --from extract)")


def cmd_ocr(args, cfg) -> None:
    """ปิดไว้: การอ่านทั้งหน้าถูกห้าม (ข้อความดิบมาจากคำสั่ง sections ขั้น ocr เท่านั้น)."""
    sys.exit(WHOLE_PAGE_BLOCKED.format(engine=args.engine))


def cmd_extract(args, cfg) -> None:
    """สกัดจากข้อความที่อ่านทีละ section แล้ว (04_ocr/<dataset>/<engine>/<วิธี>/) ไม่อ่านภาพเอง: การอ่านทั้งหน้าถูกห้าม."""
    from .pipeline import FinanceOCRPipeline

    paths, dataset = DataPaths(cfg), dataset_of(args.input)
    use_llm = {"on": True, "off": False, "auto": True, "hybrid": True}[args.llm]
    # ข้อความรายส่วนเป็น Markdown ทุก engine (.md ที่ต่อจากทุก section) จึงสกัดแบบ vlm เสมอ
    pipe = FinanceOCRPipeline(cfg, mode="vlm", engine=args.engine, use_llm=use_llm, with_reader=False,
                              strategy="hybrid" if args.llm == "hybrid" else None)
    if args.engine == "paddle":  # pipeline โหมด vlm ตั้งชื่อตามโมเดล VLM (ชื่อเดิม paddle = Typhoon) แต่ข้อความนี้มาจาก PaddleOCR จริง
        pipe.reader.name = "paddle"
    extractor = ("hybrid" if pipe.strategy == "hybrid" else "llm") if pipe.llm else "rules"
    base = Path(args.out) if args.out else None
    raw_dir = base / "04_ocr" / variant_of(args.variant) if base else paths.raw_data(args.engine, dataset, args.variant)
    final_dir = base / "05_extract" / variant_of(args.variant) / extractor if base else paths.final(args.engine, dataset, extractor, args.variant)
    settings = {
        "engine": pipe.engine_name, "variant": args.variant, "profile": args.profile,
        "keep_raw": args.keep_raw, "raw_data": Path(os.path.relpath(raw_dir, final_dir)).as_posix(),
        **({"llm_model": cfg["llm_extractor"]["model"]} if pipe.llm else {}),
    }
    with Run("extract", args.input, cfg, final_dir, settings) as run:
        all_files = list(iter_input_files(args.input))
        nest = nest_per_file(len(all_files))  # นับจากอินพุตทั้งหมด ไม่ใช่เฉพาะที่เลือกด้วย --only (โครงโฟลเดอร์ต่อไฟล์จะได้ตรงกับรอบเต็ม)
        files = [f for f in all_files if not args.only or any(f.stem.endswith(o) for o in args.only)]
        for f in files:
            t0 = time.perf_counter()
            pages = raw_pages(raw_dir, f.stem, "md")
            if not pages:
                print(f"ข้าม {f.name}: ไม่มีข้อความดิบใน {raw_dir} (รัน sections --engine {args.engine} --to ocr ก่อน)")
                continue
            for page_no, r in ((no, pipe.process_raw(str(f), no, *load_raw(p))) for no, p in pages):
                seconds = round(time.perf_counter() - t0, 2)
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
                for key, why in (pipe.last_rejected if pipe.strategy == "hybrid" else []):
                    print(f"   ⊘ LLM เสนอ {key} แต่ไม่รับ: {why}")
                t0 = time.perf_counter()
    print(f"manifest: {run.dir / 'manifest.json'}")


def cmd_sections(args, cfg) -> None:
    """flow หลักห้าขั้นของสแกนจริง: preprocess → layout → images → ocr → extract (ขั้นที่ผลยังใช้ได้ถูกข้าม ดู sections/flow.py)."""
    from .sections.build import build_images, build_layouts
    from .sections.flow import STAGES, images_are_current, layout_is_current, orientation_flags, stems_needing
    from .sections.layout import MAX_PIECE_FRAC
    from .sections.read import read_dataset

    paths, dataset = DataPaths(cfg), dataset_of(args.input)
    if args.max_piece_frac is None:
        args.max_piece_frac = MAX_PIECE_FRAC
    first, last = STAGES.index(args.start), STAGES.index(args.stop)
    if first > last:
        sys.exit(f"--from {args.start} มาหลัง --to {args.stop}")
    todo = STAGES[first:last + 1]
    parser = build_parser()
    common = ["--config", args.config] if args.config else []

    def banner(stage: str) -> None:
        print(f"\n=== [{STAGES.index(stage) + 1}/{len(STAGES)}] {stage} ===", flush=True)

    only = args.only or None
    if "preprocess" in todo:
        banner("preprocess")
        # แยก process: cuDNN ของ Paddle (ตัวตรวจทิศ) กับของ torch (OpenThai ขั้น ocr) ต่างรุ่นกัน อยู่ใน process เดียวกันไม่ได้ (ดู preprocessing/orient.py)
        import subprocess

        utf8 = ["-X", "utf8"] if sys.flags.utf8_mode else []
        cmd = [sys.executable, *utf8, "-m", "fin_extractor", *common, "preprocess", args.input, "--profile", args.profile]
        if subprocess.run(cmd).returncode:
            sys.exit("preprocess ล้ม")
    from .sections.flow import final_stems

    stems = final_stems(paths, dataset, only, args.profile)
    if not stems and set(todo) - {"preprocess"}:
        sys.exit(f"ไม่พบภาพ final ใน {paths.preprocess_final(dataset, args.profile)} — เริ่มจาก --from preprocess")
    if "layout" in todo:
        banner("layout")
        stale = stems_needing(lambda s: not args.redo and layout_is_current(paths, dataset, s, args.table_mode, args.max_piece_frac, args.profile, args.fill_aware, args.dotted_rules), stems)
        print(f"ผังใช้ได้อยู่แล้ว {len(stems) - len(stale)} ไฟล์ · ทำใหม่ {len(stale)}", flush=True)
        if stale:
            build_layouts(paths, dataset, args.profile, stale, args.table_mode, args.max_piece_frac, args.fill_aware, args.dotted_rules)
    variants = () if args.variant == "color" else (args.variant,)  # ค่าตั้งต้นเขียนเฉพาะวิธีที่ใช้ (flat/bin ทดลองผ่าน scripts/sections/make_clean_sections.py ซึ่งเขียนครบทั้งสามวิธี)
    if "images" in todo and variants:
        banner("images")
        stale = stems_needing(lambda s: not args.redo and not args.drop_empty_rows and images_are_current(paths, dataset, s, variants, args.profile), stems)
        print(f"ภาพรายส่วน ({args.variant}) ใช้ได้อยู่แล้ว {len(stems) - len(stale)} ไฟล์ · ทำใหม่ {len(stale)}", flush=True)
        if stale:
            build_images(paths, dataset, stale, args.drop_empty_rows, variants)
    if "ocr" in todo:
        banner("ocr")
        summary = read_dataset(paths, cfg, dataset, stems, args.engine, args.variant, redo=args.redo)
        print(f"รันรอบนี้ {len(summary)} ไฟล์ (ที่เหลือมีผลแล้ว ข้าม)")
    if "extract" in todo:
        banner("extract")
        a = parser.parse_args([*common, "extract", args.input, "--engine", args.engine, "--llm", "off",
                               "--variant", args.variant, "--profile", args.profile, *(["--only", *args.only] if args.only else [])])
        a.func(a, cfg)
    flags = orientation_flags(paths, dataset, stems, args.profile)
    if flags:
        print(f"\n⚠ ตรวจทิศไม่ได้ {len(flags)}/{len(stems)} ไฟล์ (ภาพอาจตะแคงโดยไม่มีสัญญาณ) ตรวจภาพใน {paths.preprocess_final(dataset, args.profile)} "
              "แล้วสั่งทิศเองได้: fin_extractor preprocess <ไฟล์> --orientation 90|180|270 (แล้วรัน sections --from layout --only <เลขไฟล์>)")
        for stem, err in flags:
            print(f"   {stem[-4:]}: {err}")
    print(f"\nผลอยู่ที่ {paths.raw_data(args.engine, dataset, args.variant)} และ {paths.final(args.engine, dataset, 'rules', args.variant)}")


def cmd_compare(args, cfg) -> None:
    """เทียบผลสกัดข้ามวิธี (luma/flat/bin) แถวที่เทียบไม่แฟร์ถูกติดธงและไม่นับในยอดรวม (ดู sections/compare.py)."""
    from .sections.compare import compare_variants, to_markdown

    paths, dataset = DataPaths(cfg), dataset_of(args.input)
    result = compare_variants(paths, dataset, args.engine, tuple(args.variants))
    text = to_markdown(result)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")


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
    s.add_argument("--orientation", type=int, choices=[0, 90, 180, 270],
                   help="สั่งมุมหมุน (องศาตามเข็ม) เองแทนตัวตรวจทิศอัตโนมัติ ใช้เมื่อตัวตรวจล้ม (ดูคำเตือน/ meta.orientation_error ใน manifest) "
                        "ทำใหม่ไฟล์นั้นทันที ไม่เก็บลง settings.json เหมาะกับสั่งทีละไฟล์")
    s.add_argument("--force", action="store_true", help="ทำใหม่ทับ ถ้าตั้งค่าเปลี่ยนจากที่เก็บไว้ = ล้างโฟลเดอร์ profile นั้นก่อน")
    s.add_argument("--out", help="โฟลเดอร์ profile เอง (ค่าตั้งต้น data/02_preprocess/<dataset>/<profile>)")
    s.set_defaults(func=cmd_preprocess)

    s = sub.add_parser("ocr", help="ปิดแล้ว: ห้ามอ่านทั้งหน้า ใช้ sections --to ocr")
    s.add_argument("input")
    s.add_argument("--engine", choices=SECTION_ENGINES, default="openthai")
    s.set_defaults(func=cmd_ocr)

    s = sub.add_parser("extract", help="สกัด field → JSON จากข้อความที่อ่านทีละ section แล้ว (ไม่อ่านภาพ)")
    s.add_argument("input")
    s.add_argument("--engine", choices=SECTION_ENGINES, default="openthai", help="อ่านข้อความจาก 04_ocr/<dataset>/<engine>/<วิธี>/")
    s.add_argument("--variant", choices=["luma", "flat", "bin", "color"], default="luma", help="ภาพรายส่วนแบบไหนที่ใช้ตอนอ่าน (โฟลเดอร์ <วิธี>)")
    s.add_argument("--llm", choices=["auto", "on", "off", "hybrid"], default="off",
                   help="off = rules อย่างเดียว | on/auto = rules และ LLM สกัดแยก เลือกชุดที่ check ดีกว่า | "
                        "hybrid = rules นำ LLM เติมเฉพาะช่องที่ยังไม่เสร็จ ค่าต้องผ่านด่านก่อนรับ (ผลลง <extractor>/hybrid/)")
    profile_arg(s)
    s.add_argument("--only", nargs="*", help="สกัดเฉพาะไฟล์ที่ชื่อลงท้ายด้วยเลขเหล่านี้ เช่น 1309 (ไม่ใส่ = ทุกไฟล์ในอินพุต)")
    s.add_argument("--keep-raw", action="store_true", help="เก็บข้อความดิบใน JSON ผลสุดท้ายด้วย")
    s.add_argument("--out", help="โฟลเดอร์ปลายทางเอง (อ่านจาก <out>/04_ocr/<วิธี>/ เขียนที่ <out>/05_extract/<วิธี>/<extractor>/)")
    s.set_defaults(func=cmd_extract)

    s = sub.add_parser("sections", help="flow หลักของสแกนจริง: preprocess → แบ่ง section → ภาพรายส่วน → OCR → สกัด (ข้ามขั้นที่ผลยังใช้ได้)")
    s.add_argument("input", help="ไฟล์หรือโฟลเดอร์อินพุต (ชื่อโฟลเดอร์ = ชื่อชุดข้อมูล)")
    s.add_argument("--variant", choices=["luma", "flat", "bin", "color"], default="luma",
                   help="ภาพรายส่วนที่ส่งเข้า OCR (ค่าตั้งต้น luma วัดแล้วดีสุดบน 18 ใบ ดู docs/03-sections.md) color = ตัดจากภาพ final สี ไม่มีขั้น images")
    s.add_argument("--engine", choices=SECTION_ENGINES, default="openthai", help="ตัวอ่านรายส่วน (typhoon ต้องเปิด Ollama)")
    s.add_argument("--from", dest="start", choices=["preprocess", "layout", "images", "ocr", "extract"], default="preprocess",
                   help="เริ่มที่ขั้นนี้ (ขั้นก่อนหน้าไม่รัน ใช้ผลที่มี)")
    s.add_argument("--to", dest="stop", choices=["preprocess", "layout", "images", "ocr", "extract"], default="extract", help="จบที่ขั้นนี้")
    s.add_argument("--only", nargs="*", help="เลขท้ายชื่อไฟล์ เช่น 1309")
    s.add_argument("--redo", action="store_true", help="ทำใหม่ทุกขั้นที่รัน แม้ผลเดิมยังใช้ได้")
    s.add_argument("--table-mode", choices=["whole", "cells", "bands", "rows"], default="rows", help="วิธีแบ่งตาราง (ดู docs/03-sections.md)")
    s.add_argument("--max-piece-frac", type=float, default=None,
                   help="ส่วนที่สูงกว่านี้ (สัดส่วนความสูงหน้า) ถูกตัดเป็นท่อน 0 = ไม่ตัด (ไม่ใส่ = ค่าใน sections/layout.py)")
    s.add_argument("--fill-aware", action="store_true",
                   help="ผังนับตัวอักษรบนพื้นสีเป็นหมึกและไม่ตัดผ่านแถบสี (กัน OCR เดาข้อความจากเสี้ยวแถบ เช่น 1306) ผลวัด 6 ใบที่ผังเปลี่ยน: ฟิลด์ที่ขาดรวมเท่าเดิม (ดู docs/03-sections.md) จึงไม่เปิดเป็นค่าตั้งต้น")
    s.add_argument("--dotted-rules", action="store_true",
                   help="ผังตัดตามเส้นแบ่งแถวที่เป็นเส้นประด้วย (ฟอร์มอย่าง 50 ทวิ ที่เส้นแถวเป็นเส้นประ) ยังไม่ได้วัดกับ OCR จึงไม่เปิดเป็นค่าตั้งต้น (ดู docs/03-sections.md)")
    s.add_argument("--drop-empty-rows", action="store_true", help="แบบ bin เท่านั้น: ลบแถวตารางที่ช่องตัวเลขว่าง (ดู docs/03-sections.md)")
    profile_arg(s)
    s.set_defaults(func=cmd_sections)

    s = sub.add_parser("compare", help="เทียบผลสกัดข้ามวิธี luma/flat/bin (ติดธงแถวที่เทียบไม่แฟร์)")
    s.add_argument("input", help="โฟลเดอร์อินพุต (ใช้เอาชื่อชุดข้อมูล)")
    s.add_argument("--engine", default="openthai")
    s.add_argument("--variants", nargs="+", default=["luma", "flat", "bin"], choices=["luma", "flat", "bin", "color"])
    s.add_argument("--out", help="เขียนตารางเป็นไฟล์ .md ด้วย")
    s.set_defaults(func=cmd_compare)

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
