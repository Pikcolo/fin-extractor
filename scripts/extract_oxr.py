"""รัน pipeline เต็มโดยใช้ OXR เป็นตัวอ่านข้อความ (เทียบกับ ocr-rules / vlm-llm ที่ใช้อินพุตและขั้นสกัดเดียวกัน)

ภาพผ่าน loader + preprocess ของโปรเจกต์เหมือนที่ Paddle/Typhoon ได้ แล้วส่ง JPEG ไปที่ OXR **ในเครื่อง** (container `oxr` ที่ 127.0.0.1:8000)
ผลเป็น Markdown + ตาราง HTML (เหมือน Typhoon) จึงเข้าทางเดียวกัน: markdown_to_lines → rules → validation
ไม่มี LLM ตัวสกัด และไม่มี finish_reason จาก OXR ผลวนซ้ำ/ถูกตัดจึงไม่ถูกตั้งธง ``vlm_complete``

ห้ามชี้ ``--api`` ไปที่ free API กับสแกนจริง (ข้อมูลออกนอกเครื่อง) สคริปต์จึงปฏิเสธโฮสต์ที่ไม่ใช่ localhost

    .venv/Scripts/python scripts/extract_oxr.py data/raw/finance_doc_scan
    # ผล: data/output/extract/<อินพุต>__oxr-rules/
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import cv2
import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fin_extractor.config import load_config  # noqa: E402
from fin_extractor.io import iter_input_files  # noqa: E402
from fin_extractor.pipeline import FinanceOCRPipeline  # noqa: E402
from fin_extractor.runs import Run  # noqa: E402


@dataclass
class OXRResult:
    engine: str
    markdown: str
    truncated: bool = False


class OXRReader:
    name = "oxr"

    def __init__(self, api: str, timeout: int = 900):
        self.api, self.timeout = api, timeout

    def recognize(self, img) -> OXRResult:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        r = httpx.post(
            f"{self.api}/v1/parse/sync",
            files=[("file", ("page.jpg", buf.tobytes(), "image/jpeg")), ("output_formats", (None, "markdown")),
                   ("api_key", (None, "dummy"))],
            timeout=self.timeout,
        )
        r.raise_for_status()
        body = r.json()
        if body.get("code") != 0 or not body.get("data"):
            raise RuntimeError(f"OXR error: {body.get('code')} {body.get('message')}")
        return OXRResult("oxr", body["data"]["result"]["outputs"]["markdown"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--out")
    args = ap.parse_args()
    if urlparse(args.api).hostname not in ("127.0.0.1", "localhost"):
        sys.exit("ปฏิเสธ: --api ต้องเป็น localhost (ไม่ส่งสแกนจริงออกนอกเครื่อง)")

    cfg = load_config()
    pipe = FinanceOCRPipeline(cfg, mode="vlm", use_llm=False)  # ขั้นสกัด/ตรวจเหมือนทางของ Typhoon แบบ rules
    pipe.reader = OXRReader(args.api)
    settings = {"mode": "oxr", "engine": pipe.engine_name, "api": args.api}
    with Run("extract", args.input, "oxr-rules", cfg, args.out, settings) as run:
        results = run.sub("results")
        for f in iter_input_files(args.input):
            t0 = time.perf_counter()
            for r in pipe.process_file(f):
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
                      f"missing={','.join(r.missing_fields) or '-'} ({seconds:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
