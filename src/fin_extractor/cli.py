"""Command line: ``python -m fin_extractor <command> ...``

  preprocess  ปรับภาพแล้วเก็บภาพสุดท้ายไว้ให้ทุก engine ใช้ร่วมกัน (ดูผลทีละขั้นด้วย --steps)
  ocr         อ่านข้อความดิบด้วย OCR engine หรือ VLM แล้วเก็บที่ 04_ocr
  extract     pipeline เต็ม → JSON ตาม schema (--reuse-raw = สกัดซ้ำจากข้อความดิบที่เก็บไว้ ไม่เรียก OCR)
  schema      export JSON Schema

ผลอยู่ใต้ data/ ตามโครงใน fin_extractor/paths.py:
  02_preprocess/<dataset>/<profile>/{final,steps}/   04_ocr/<dataset>/<engine>/<วิธี>/   05_extract/<dataset>/<engine>/<วิธี>/<extractor>/
  (วิธี = page อ่านทั้งหน้า · luma|flat|bin|color อ่านทีละ section) ขั้น 03_sections ทำด้วย scripts/ (ดู docs/14)
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
from .paths import DEFAULT_PROFILE, FINAL_DIR, DataPaths, dataset_of, doc_dir, find_dir, nest_per_file, variant_of
from .runs import Run
from .stages import load_raw, prepared_pages, raw_pages, save_raw
from .vlm import VLM_ENGINES, create_reader, vlm_model


def engine_label(mode: str, engine: str) -> str:
    """ชื่อโฟลเดอร์ engine ใน 04_ocr/ และ 05_extract/ (mode vlm: --engine paddle/vlm = typhoon, openthai = openthai)."""
    return VLM_ENGINES.get(engine, "typhoon") if mode == "vlm" else engine


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


def cmd_ocr(args, cfg) -> None:
    paths, dataset = DataPaths(cfg), dataset_of(args.input)
    if args.engine in VLM_ENGINES:
        reader = create_reader(args.engine, cfg)
    else:
        from .ocr import create_engine

        reader = create_engine(args.engine, cfg)

    mode = "vlm" if args.engine in VLM_ENGINES else "ocr"
    label = engine_label(mode, args.engine)
    ext = "md" if mode == "vlm" else "txt"
    raw_dir = Path(args.out) / "04_ocr" / variant_of(None) if args.out else paths.raw_data(label, dataset)
    pdir = paths.preprocess(dataset, args.profile)
    settings = {"engine": label, "profile": args.profile, **({"model": vlm_model(args.engine, cfg)} if mode == "vlm" else {})}
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
    use_llm = {"on": True, "off": False, "auto": None, "hybrid": True}[args.llm]
    pipe = FinanceOCRPipeline(cfg, mode=args.mode, engine=args.engine, use_llm=use_llm, with_reader=not args.reuse_raw,
                              strategy="hybrid" if args.llm == "hybrid" else None)
    label = engine_label(args.mode, args.engine)
    extractor = ("hybrid" if pipe.strategy == "hybrid" else "llm") if pipe.llm else "rules"
    ext = "md" if args.mode == "vlm" else "txt"
    if args.variant and not args.reuse_raw:
        sys.exit("--variant ใช้กับ --reuse-raw เท่านั้น (อ่านข้อความดิบของ scripts/run_sections.py --variant ... ที่เก็บไว้ ไม่ใช่การอ่านสด)")
    base = Path(args.out) if args.out else None  # raw_data_<แบบ>/ และ final_<แบบ>/ ข้างผลปกติ ไม่ทับกัน
    raw_dir = base / "04_ocr" / variant_of(args.variant) if base else paths.raw_data(label, dataset, args.variant)
    final_dir = base / "05_extract" / variant_of(args.variant) / extractor if base else paths.final(label, dataset, extractor, args.variant)
    pdir = paths.preprocess(dataset, args.profile)
    settings = {
        "mode": args.mode, "engine": pipe.engine_name, "profile": args.profile, "reuse_raw": args.reuse_raw,
        "keep_raw": args.keep_raw, "raw_data": Path(os.path.relpath(raw_dir, final_dir)).as_posix(),
        **({"vlm_model": vlm_model(args.engine, cfg)} if args.mode == "vlm" else {}),
        **({"llm_model": cfg["llm_extractor"]["model"]} if pipe.llm else {}),
    }
    with Run("extract", args.input, cfg, final_dir, settings) as run:
        all_files = list(iter_input_files(args.input))
        nest = nest_per_file(len(all_files))  # นับจากอินพุตทั้งหมด ไม่ใช่เฉพาะที่เลือกด้วย --only (โครงโฟลเดอร์ต่อไฟล์จะได้ตรงกับรอบเต็ม)
        files = [f for f in all_files if not args.only or any(f.stem.endswith(o) for o in args.only)]
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
        a = parser.parse_args([*common, "preprocess", args.input, "--profile", args.profile])
        a.func(a, cfg)
    from .sections.flow import final_stems

    stems = final_stems(paths, dataset, only, args.profile)
    if not stems and set(todo) - {"preprocess"}:
        sys.exit(f"ไม่พบภาพ final ใน {paths.preprocess_final(dataset, args.profile)} — เริ่มจาก --from preprocess")
    if "layout" in todo:
        banner("layout")
        stale = stems_needing(lambda s: not args.redo and layout_is_current(paths, dataset, s, args.table_mode, args.max_piece_frac, args.profile, args.fill_aware), stems)
        print(f"ผังใช้ได้อยู่แล้ว {len(stems) - len(stale)} ไฟล์ · ทำใหม่ {len(stale)}", flush=True)
        if stale:
            build_layouts(paths, dataset, args.profile, stale, args.table_mode, args.max_piece_frac, args.fill_aware)
    variants = () if args.variant == "color" else (args.variant,)  # ค่าตั้งต้นเขียนเฉพาะวิธีที่ใช้ (flat/bin ทดลองผ่าน scripts/make_clean_sections.py)
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
        if args.engine != "openthai":
            print(f"ข้ามขั้น extract: ยังไม่รองรับ engine {args.engine} (รองรับ openthai)")
            return
        a = parser.parse_args([*common, "extract", args.input, "--mode", "vlm", "--engine", args.engine, "--llm", "off", "--reuse-raw",
                               "--variant", args.variant, "--profile", args.profile, *(["--only", *args.only] if args.only else [])])
        a.func(a, cfg)
    flags = orientation_flags(paths, dataset, stems, args.profile)
    if flags:
        print(f"\n⚠ ตรวจทิศไม่ได้ {len(flags)}/{len(stems)} ไฟล์ (ภาพอาจตะแคงโดยไม่มีสัญญาณ) ตรวจภาพใน {paths.preprocess_final(dataset, args.profile)} "
              "แล้วสั่งทิศเองได้: fin_extractor preprocess <ไฟล์> --orientation 90|180|270 (แล้วรัน sections --from layout --only <เลขไฟล์>)")
        for stem, err in flags:
            print(f"   {stem[-4:]}: {err}")
    print(f"\nผลอยู่ที่ {paths.raw_data(args.engine, dataset, args.variant)} และ {paths.final(args.engine, dataset, 'rules', args.variant)}")


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
    s.add_argument("--out", help="โฟลเดอร์ profile เอง (ค่าตั้งต้น data/preprocess/<dataset>/<profile>)")
    s.set_defaults(func=cmd_preprocess)

    s = sub.add_parser("ocr", help="อ่านข้อความดิบ → raw_data")
    s.add_argument("input")
    s.add_argument("--engine", choices=[*ENGINES, *VLM_ENGINES], default="paddle")
    profile_arg(s)
    s.add_argument("--out", help="โฟลเดอร์ปลายทางเอง (ค่าตั้งต้น data/04_ocr/<dataset>/<engine>/ และ data/05_extract/<dataset>/<engine>/) ข้อความดิบอยู่ใต้ raw_data/")
    s.set_defaults(func=cmd_ocr)

    s = sub.add_parser("extract", help="สกัด field → JSON")
    s.add_argument("input")
    s.add_argument("--mode", choices=["ocr", "vlm"], default="ocr")
    s.add_argument("--engine", choices=[*ENGINES, "openthai"], default="paddle",
                   help="--mode ocr: paddle | --mode vlm: ค่าตั้งต้น (paddle) = Typhoon, openthai = OpenThai 2.0 OCR")
    s.add_argument("--llm", choices=["auto", "on", "off", "hybrid"], default="auto",
                   help="auto = ใช้ LLM เฉพาะ mode vlm | on = rules และ LLM สกัดแยก เลือกชุดที่ check ดีกว่า | "
                        "hybrid = rules นำ LLM เติมเฉพาะช่องที่ยังไม่เสร็จ ค่าต้องผ่านด่านก่อนรับ (ผลลง final/hybrid/)")
    profile_arg(s)
    s.add_argument("--reuse-raw", action="store_true",
                   help="สกัดซ้ำจากข้อความดิบที่เก็บไว้ใน raw_data (ไม่โหลด/เรียก OCR หรือ VLM ใช้ลองเปลี่ยน rules/LLM)")
    s.add_argument("--variant", choices=["luma", "flat", "bin", "color"],
                   help="ใช้กับ --reuse-raw: อ่านข้อความดิบจาก raw_data_<แบบ>/ (ผลของ scripts/run_sections.py --variant) แล้วเขียนผลที่ final_<แบบ>/<extractor>/ "
                        "ไม่ใส่ = raw_data/ และ final/ ตามปกติ")
    s.add_argument("--only", nargs="*", help="สกัดเฉพาะไฟล์ที่ชื่อลงท้ายด้วยเลขเหล่านี้ เช่น 1309 (ไม่ใส่ = ทุกไฟล์ในอินพุต)")
    s.add_argument("--keep-raw", action="store_true", help="เก็บข้อความดิบใน JSON ผลสุดท้ายด้วย")
    s.add_argument("--out", help="โฟลเดอร์ปลายทางเอง (ค่าตั้งต้น data/04_ocr/<dataset>/<engine>/ และ data/05_extract/<dataset>/<engine>/) ผลอยู่ใต้ raw_data/ และ final/<extractor>/")
    s.set_defaults(func=cmd_extract)

    s = sub.add_parser("sections", help="flow หลักของสแกนจริง: preprocess → แบ่ง section → ภาพรายส่วน → OCR → สกัด (ข้ามขั้นที่ผลยังใช้ได้)")
    s.add_argument("input", help="ไฟล์หรือโฟลเดอร์อินพุต (ชื่อโฟลเดอร์ = ชื่อชุดข้อมูล)")
    s.add_argument("--variant", choices=["luma", "flat", "bin", "color"], default="luma",
                   help="ภาพรายส่วนที่ส่งเข้า OCR (ค่าตั้งต้น luma วัดแล้วดีสุดบน 18 ใบ ดู docs/14) color = ตัดจากภาพ final สี ไม่มีขั้น images")
    s.add_argument("--engine", choices=["openthai", "oxr"], default="openthai", help="ตัวอ่านรายส่วน (ขั้น extract รองรับ openthai)")
    s.add_argument("--from", dest="start", choices=["preprocess", "layout", "images", "ocr", "extract"], default="preprocess",
                   help="เริ่มที่ขั้นนี้ (ขั้นก่อนหน้าไม่รัน ใช้ผลที่มี)")
    s.add_argument("--to", dest="stop", choices=["preprocess", "layout", "images", "ocr", "extract"], default="extract", help="จบที่ขั้นนี้")
    s.add_argument("--only", nargs="*", help="เลขท้ายชื่อไฟล์ เช่น 1309")
    s.add_argument("--redo", action="store_true", help="ทำใหม่ทุกขั้นที่รัน แม้ผลเดิมยังใช้ได้")
    s.add_argument("--table-mode", choices=["whole", "cells", "bands", "rows"], default="rows", help="วิธีแบ่งตาราง (ดู docs/14)")
    s.add_argument("--max-piece-frac", type=float, default=None,
                   help="ส่วนที่สูงกว่านี้ (สัดส่วนความสูงหน้า) ถูกตัดเป็นท่อน 0 = ไม่ตัด (ไม่ใส่ = ค่าใน sections/layout.py)")
    s.add_argument("--fill-aware", action="store_true",
                   help="ผังนับตัวอักษรบนพื้นสีเป็นหมึกและไม่ตัดผ่านแถบสี (กัน OCR เดาข้อความจากเสี้ยวแถบ เช่น 1306) ผลวัด 6 ใบที่ผังเปลี่ยน: ฟิลด์ที่ขาดรวมเท่าเดิม (ดู docs/14) จึงไม่เปิดเป็นค่าตั้งต้น")
    s.add_argument("--drop-empty-rows", action="store_true", help="แบบ bin เท่านั้น: ลบแถวตารางที่ช่องตัวเลขว่าง (ดู docs/14)")
    profile_arg(s)
    s.set_defaults(func=cmd_sections)

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
