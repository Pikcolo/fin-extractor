"""สรุปผล ``extract`` หลายโฟลเดอร์ (เช่น rules / llm / hybrid) เป็นตาราง Markdown เทียบกันได้ ไม่ต้องมีเฉลย

ตัวชี้วัดชุดเดียวกับตารางใน docs/05-extraction.md §5D: สถานะ · ฟิลด์ขาด (review = missing) · check ผ่าน/ไม่ผ่าน
· ค่าที่ไม่พบในข้อความ (เหตุผล "ไม่ปรากฏในข้อความ") · ฟิลด์ conflict · ฟิลด์ที่ได้จาก LLM (review.source = llm) · ค่าที่ด่านปัดตก (rejected)
บอกแค่ความสอดคล้องภายใน ไม่ใช่ความถูกต้อง

    .venv/Scripts/python -X utf8 scripts/eval/summarize_extract.py <โฟลเดอร์ผล>... [--detail]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

NOT_IN_TEXT = "ไม่ปรากฏในข้อความ"


def load(folder: Path) -> dict[str, dict]:
    return {p.stem.split("_")[-2]: json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(folder.rglob("*_p1.json"))}


def stats(r: dict) -> dict:
    rv = r.get("review") or {}
    return {
        "status": r["status"],
        "missing": sorted(k for k, v in rv.items() if v["level"] == "missing"),
        "pass": sum(c["passed"] is True for c in r["checks"]),
        "fail": sum(c["passed"] is False for c in r["checks"]),
        "not_in_text": sorted(k for k, v in rv.items() if any(NOT_IN_TEXT in x for x in v["reasons"])),
        "conflict": sum(v["level"] == "conflict" for v in rv.values()),
        "llm": sorted(k for k, v in rv.items() if v.get("source") == "llm"),
        "rejected": len(r.get("rejected") or []),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+", type=Path)
    ap.add_argument("--detail", action="store_true", help="ตารางรายใบ: สถานะ + ฟิลด์ที่ขาด")
    args = ap.parse_args()
    runs = {f.name: {k: stats(v) for k, v in load(f).items()} for f in args.folders}
    print("| ชุด | ใบ | สถานะ | ฟิลด์ขาด | check ผ่าน/ไม่ผ่าน | ค่าที่ไม่พบในข้อความ | conflict | ฟิลด์จาก LLM | ด่านปัดตก |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, docs in runs.items():
        s = docs.values()
        st = Counter(x["status"] for x in s)
        print(f"| {name} | {len(docs)} | {' · '.join(f'{n} {k}' for k, n in st.most_common())} | {sum(len(x['missing']) for x in s)} "
              f"| {sum(x['pass'] for x in s)} / {sum(x['fail'] for x in s)} | {sum(len(x['not_in_text']) for x in s)} "
              f"| {sum(x['conflict'] for x in s)} | {sum(len(x['llm']) for x in s)} | {sum(x['rejected'] for x in s)} |")
    if args.detail:
        names = list(runs)
        print("\n| ใบ | " + " | ".join(names) + " |")
        print("|---|" + "---|" * len(names))
        for doc in sorted(set().union(*runs.values())):
            cells = []
            for n in names:
                x = runs[n].get(doc)
                cells.append("—" if x is None else f"{x['status']}" + (f" · ขาด {', '.join(x['missing'])}" if x["missing"] else "")
                             + (f" · ไม่พบในข้อความ {', '.join(x['not_in_text'])}" if x["not_in_text"] else ""))
            print(f"| {doc} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
