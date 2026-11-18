"""ย้ายข้อมูลเก่าไปโครงโฟลเดอร์ใหม่ที่เรียงตามขั้น (ดู ``src/fin_extractor/paths.py``) **ย้ายอย่างเดียว ไม่ลบ ไม่ทับ**

    python -X utf8 scripts/migrate_data_layout.py                 # dry-run: พิมพ์แผนเก่า → ใหม่ ยังไม่ย้ายอะไร
    python -X utf8 scripts/migrate_data_layout.py --apply         # ย้ายจริง
    python -X utf8 scripts/migrate_data_layout.py --data-dir D:/x/data --apply

แผน (ใต้ data/)
    raw/<x>                                              → 01_input/<x>
    preprocess/<ds>/<profile>/{final,steps,*.json}       → 02_preprocess/<ds>/<profile>/…
    preprocess/<ds>/<profile>/sections/features          → 03_sections/<ds>/<profile>/layout
    preprocess/<ds>/<profile>/sections/clean/<วิธี>      → 03_sections/<ds>/<profile>/images/<วิธี>
    preprocess/<ds>/<profile>/sections/{router,ocr}      → 03_sections/<ds>/<profile>/{router,ocr_images}
    result_ocr/<engine>/<ds>/raw_data                    → 04_ocr/<ds>/<engine>/page   (มี *.sections.json = อ่านทีละ section จากภาพสี → color)
    result_ocr/<engine>/<ds>/raw_data_<วิธี>             → 04_ocr/<ds>/<engine>/<วิธี>
    result_ocr/<engine>/<ds>/visualize                   → 04_ocr/<ds>/<engine>/visualize
    result_ocr/<engine>/<ds>/final/<extractor>           → 05_extract/<ds>/<engine>/page/<extractor>
    result_ocr/<engine>/<ds>/final_<วิธี>/<extractor>    → 05_extract/<ds>/<engine>/<วิธี>/<extractor>

กติกา: ปลายทางมีอยู่แล้ว = ข้าม (รายงานเป็น conflict) · ต้นทางไม่มี = ข้าม · รันซ้ำได้ (ไม่มีอะไรให้ย้ายแล้วก็ไม่ทำอะไร)
โฟลเดอร์เก่าที่ว่างหลังย้ายถูกลบด้วย rmdir (ลบได้เฉพาะโฟลเดอร์ว่าง) ``manifest.json`` ของผลสกัดยังบันทึกพาธเก่าใน settings.raw_data
รัน ``extract --reuse-raw --variant <วิธี>`` ใหม่เพื่อเขียน manifest ให้ตรง
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fin_extractor.config import load_config  # noqa: E402
from fin_extractor.paths import DEFAULT_FOLDERS, DataPaths  # noqa: E402


def _kids(p: Path) -> list[Path]:
    return sorted(p.iterdir()) if p.is_dir() else []


def plan(root: Path) -> list[tuple[Path, Path]]:
    """[(ต้นทาง, ปลายทาง)] ตามแผนข้างบน (เฉพาะที่ต้นทางมีอยู่จริง)."""
    new = {k: root / v for k, v in DEFAULT_FOLDERS.items()}
    out: list[tuple[Path, Path]] = []

    for child in _kids(root / "raw"):
        out.append((child, new["raw"] / child.name))

    for ds in _kids(root / "preprocess"):
        for prof in _kids(ds):
            if not prof.is_dir():
                continue
            for item in _kids(prof):
                if item.name == "sections":
                    continue
                out.append((item, new["preprocess"] / ds.name / prof.name / item.name))
            sec = prof / "sections"
            base = new["sections"] / ds.name / prof.name
            if (sec / "features").exists():
                out.append((sec / "features", base / "layout"))
            for v in _kids(sec / "clean"):
                out.append((v, base / "images" / v.name))
            if (sec / "router").exists():
                out.append((sec / "router", base / "router"))
            if (sec / "ocr").exists():
                out.append((sec / "ocr", base / "ocr_images"))

    for eng in _kids(root / "result_ocr"):
        for ds in _kids(eng):
            if not ds.is_dir():
                continue
            ocr, ext = new["ocr"] / ds.name / eng.name, new["extract"] / ds.name / eng.name
            for item in _kids(ds):
                n = item.name
                if n == "raw_data":
                    by_section = any(item.glob("*/*.sections.json")) or any(item.glob("*.sections.json"))
                    out.append((item, ocr / ("color" if by_section else "page")))
                elif n.startswith("raw_data_"):
                    out.append((item, ocr / n.removeprefix("raw_data_")))
                elif n == "visualize":
                    out.append((item, ocr / "visualize"))
                elif n == "final":
                    out.extend((x, ext / "page" / x.name) for x in _kids(item))
                elif n.startswith("final_"):
                    out.extend((x, ext / n.removeprefix("final_") / x.name) for x in _kids(item))
    return out


def apply(moves: list[tuple[Path, Path]]) -> tuple[list, list]:
    """ย้ายตามแผน คืน (ที่ย้ายแล้ว, conflict) ไม่ทับปลายทางที่มีอยู่."""
    done, conflicts = [], []
    for src, dst in moves:
        if dst.exists():
            conflicts.append((src, dst))
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        done.append((src, dst))
    return done, conflicts


def prune_empty(root: Path) -> list[Path]:
    """rmdir โฟลเดอร์เก่าที่ว่างแล้ว (ลึกสุดก่อน) ไม่ลบอะไรที่มีไฟล์."""
    removed = []
    for top in ("raw", "preprocess", "result_ocr"):
        base = root / top
        if not base.exists():
            continue
        for d in sorted((p for p in base.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            if not any(d.iterdir()):
                d.rmdir()
                removed.append(d)
        if not any(base.iterdir()):
            base.rmdir()
            removed.append(base)
    return removed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="ย้ายจริง (ไม่ใส่ = dry-run)")
    ap.add_argument("--data-dir", help="โฟลเดอร์ data (ค่าตั้งต้นจาก config paths.data_dir)")
    args = ap.parse_args()
    root = Path(args.data_dir) if args.data_dir else DataPaths(load_config()).root
    moves = plan(root)
    if not moves:
        print(f"ไม่มีอะไรต้องย้ายใต้ {root}")
        return
    for src, dst in moves:
        print(f"{'ย้าย' if args.apply else 'จะย้าย'}  {src.relative_to(root).as_posix()}  →  {dst.relative_to(root).as_posix()}")
    if not args.apply:
        print(f"\ndry-run: {len(moves)} รายการ (ใส่ --apply เพื่อย้ายจริง)")
        return
    done, conflicts = apply(moves)
    removed = prune_empty(root)
    print(f"\nย้ายแล้ว {len(done)} · ข้าม (ปลายทางมีอยู่แล้ว) {len(conflicts)} · ลบโฟลเดอร์ว่าง {len(removed)}")
    for src, dst in conflicts:
        print(f"  conflict: {src.relative_to(root).as_posix()} → {dst.relative_to(root).as_posix()} (ไม่ทับ)")


if __name__ == "__main__":
    main()
