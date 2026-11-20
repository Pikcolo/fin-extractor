# fin_extractor

OCR + Information Extraction สำหรับเอกสารการเงินไทย (ใบกำกับภาษี / ใบเสร็จ / invoice) ที่ **ไม่มี pattern ตายตัว**
ผลลัพธ์เป็น JSON มาตรฐานตาม [`schemas/invoice.schema.json`](schemas/invoice.schema.json) (schema 3.0: ยอดเงินเป็น `Decimal`
ระดับหลักฐานรายฟิลด์ `field_evidence`, ฟิลด์เฉพาะชนิดเอกสาร `document_fields` ตามโปรไฟล์ และ `remark` สำหรับข้อมูลที่ไม่มีช่อง)
อธิบายขั้นตอนการทำงานแบบละเอียดอยู่ใน [`docs/`](docs/README.md)

```
ไฟล์ (jpg/png/pdf)
  └─ io/loader          โหลด + render PDF เป็นภาพทีละหน้า
      └─ preprocessing  upscale → orientation → deskew   (คงสี; denoise/CLAHE/binarize ปิดเป็นค่าตั้งต้น)
          ├─ ocr/       PaddleOCR                         (mode=ocr)   → ข้อความ + bbox + confidence
          ├─ vlm/       Typhoon OCR → Markdown/HTML table  (mode=vlm)
          └─ sections/  flow หลัก: แบ่งเป็นส่วน → ภาพรายส่วน → OpenThai ทีละส่วน → .md + chunks.json → rules   (ดู "flow หลัก")
              └─ extraction/  rules (regex+keyword)  และ/หรือ  LLM → InvoiceData
                  └─ validation/  checksum เลขภาษี, Σitems, ส่วนลด, ก่อนVAT+VAT=รวม, VAT 7% → status + field_evidence
```

## เริ่มใช้งาน

ต้องมี Python 3.10 ขึ้นไป รายละเอียดทุกขั้น (GPU/CPU, Ollama, ปัญหาที่พบบ่อย) อยู่ใน [docs/06-usage.md](docs/06-usage.md)

```powershell
git clone https://github.com/Pikcolo/fin-extractor.git
cd fin-extractor
python -m venv .venv
.venv\Scripts\activate

# paddlepaddle ต้องลงก่อน เลือกอย่างใดอย่างหนึ่ง
pip install paddlepaddle-gpu==3.4.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu130/ --extra-index-url https://pypi.org/simple   # มี GPU
pip install paddlepaddle                                                                                                                  # ไม่มี GPU

pip install -e ".[all]"
pytest                                    # ต้องผ่านทั้งหมด (ไม่ต้องใช้ GPU / Ollama / ข้อมูล)
```

### รันครั้งแรก

เอกสารจริงไม่อยู่ใน git (ทุกอย่างใต้ `data/` ถูก `.gitignore`) วางไฟล์ของคุณที่ `data/01_input/` หรือสร้างชุดสังเคราะห์:

```powershell
python scripts/make_synthetic_samples.py                  # ใบกำกับภาษีสมมติ 21 ใบ → data/01_input/synthetic/
python -X utf8 -m fin_extractor extract data/01_input/synthetic   # Paddle + rules → data/05_extract/synthetic/paddle/page/rules/
```

### คำสั่งหลัก

```powershell
python -m fin_extractor preprocess data/01_input --steps           # ดูผล preprocess ทีละขั้น
python -m fin_extractor ocr data/01_input/BITKUB-001.jpg           # ข้อความดิบ (--engine vlm = Typhoon OCR)
python -m fin_extractor extract data/01_input                      # pipeline เต็ม → JSON (ค่าตั้งต้น --mode ocr)
python -m fin_extractor extract data/01_input --mode vlm           # Typhoon OCR + LLM + rules (ต้องเปิด Ollama)
python -m fin_extractor extract data/01_input --reuse-raw --only 1306   # สกัดซ้ำจากข้อความดิบ (ไม่เรียก OCR/GPU) เฉพาะไฟล์ที่ชื่อลงท้าย 1306
python -m fin_extractor sections data/01_input/finance_doc_scan    # flow หลักของสแกนจริง (ดูหัวข้อถัดไป)
python -m fin_extractor compare data/01_input/finance_doc_scan     # เทียบผลสกัด luma/flat/bin ติดธงแถวที่เทียบไม่แฟร์
python -m fin_extractor schema                                # export JSON Schema (หลังแก้ schema.py)
pytest
```

โหมด `vlm` ต้อง `ollama pull scb10x/typhoon-ocr1.5-3b` และ `ollama pull qwen2.5:3b` (หรือใช้ Typhoon API ตั้งค่าใน `.env`)

### flow หลัก: แบ่ง section แล้วอ่านด้วย OpenThai (สแกนจริง)

อินพุต → preprocess → **แบ่งเป็นส่วน (เก็บผังไว้ดู/debug)** → ภาพแต่ละส่วนที่เตรียมก่อน OCR → OCR ทีละส่วน → **ข้อความ + ที่มาของแต่ละท่อน** → สกัดเป็น schema (ข้อมูลสำคัญที่ไม่มีช่องไปอยู่ใน `remark`)
ไม่ใช้ OCR คัดส่วน ตารางเป็นก้อนเดียว ตัดท่อนสูงไม่เกิน 0.3 ของหน้า ใช้ GPU เฉพาะขั้น 4 ([docs/13](docs/13-section-preprocess.md))

```powershell
python -X utf8 -m fin_extractor sections data/01_input/finance_doc_scan            # รันครบ 5 ขั้น ขั้นที่ผลยังใช้ได้ถูกข้าม
python -X utf8 -m fin_extractor sections data/01_input/finance_doc_scan --from ocr --only 1306   # เริ่มกลางทาง/เฉพาะบางไฟล์
```

| ขั้น | ผลที่ดูได้ (ใต้ `data/`) | ใช้อะไรดู debug |
|---|---|---|
| 1 preprocess | `02_preprocess/<ชุด>/<profile>/final/` ภาพทั้งหน้าตรงแล้ว | ภาพ + `manifest.json` (`orientation_error`) |
| 2 layout | `03_sections/<ชุด>/<profile>/layout/<ไฟล์>.json` + `_layout.png` | ภาพกรอบสี + เลข id ของแต่ละส่วน |
| 3 images | `03_sections/<ชุด>/<profile>/images/luma/<ไฟล์>/sNN.png` | ภาพที่ส่งเข้า OCR จริง |
| 4 ocr | `04_ocr/<ชุด>/openthai/luma/<ไฟล์>/` `.md` `.sections.json` `.chunks.json` | `chunks.json` = ท่อนข้อความ ↔ ภาพ section ↔ บรรทัดใน `.md` |
| 5 extract | `05_extract/<ชุด>/openthai/luma/rules/<ไฟล์>_p1.json` | `field_evidence`, `remark[].source_chunk` ชี้กลับไปที่ section |

- **ค่าตั้งต้นคือ `luma`**: เทียบ 3 แบบบน 18 ใบ (ไม่มีเฉลย วัดแค่ความครบ) luma ขาดฟิลด์รวม 14, flat 16, bin (ตัวขาวบนดำ) 28 ([docs/13](docs/13-section-preprocess.md)) `--variant flat|bin|color` ผลแยกโฟลเดอร์ ไม่ทับกัน
- รันทีละขั้นได้ด้วยสคริปต์เดิม (`scripts/make_section_layouts.py` `make_clean_sections.py` `run_sections.py` เป็นตัวห่อบางๆ ของ `src/fin_extractor/sections/`)
- เทียบผลสกัดข้ามวิธี: `python -m fin_extractor compare <input> [--variants luma flat bin] [--out ไฟล์.md]` ติดธงแถวที่เทียบไม่แฟร์ (ผังต่างกัน / ผลเก่ากว่าผัง / `--drop-empty-rows` / ไม่มีผล) และไม่นับในยอดรวม
- ข้อมูลภาษีจริงไม่ถูก commit และผลวัดทั้งหมดไม่ใช่คะแนนความถูกต้อง
- ข้อควรระวัง: ตรวจ `orientation_error` ใน `manifest.json` ของ preprocess ก่อนเชื่อผล (ตัวตรวจทิศอาจล้มเงียบ)

### ผลลัพธ์อยู่ที่ไหน

ข้อมูลทั้งหมดอยู่ใต้ `data/` (ไม่ commit) ชื่อโฟลเดอร์เรียงตามขั้น ทุกขั้นใช้ `<ชุด>` และ `<ไฟล์>` เดียวกัน ตามไฟล์เดียวข้ามขั้นได้ด้วยชื่อเดียว:

```
data/
├── 01_input/<ชุด>/                              ไฟล์อินพุต
├── 02_preprocess/<ชุด>/<profile>/{final,steps}  ภาพสุดท้ายที่ทุกขั้นถัดไปใช้ร่วมกัน · ภาพทีละขั้น
├── 03_sections/<ชุด>/<profile>/                 แบ่งเป็นส่วน: layout/ (ผัง + ภาพตรวจ) · images/<luma|flat|bin>/<ไฟล์>/ (ภาพรายส่วน)
├── 04_ocr/<ชุด>/<engine>/<วิธี>/<ไฟล์>/          ข้อความจาก OCR (วิธี = page อ่านทั้งหน้า · luma|flat|bin|color อ่านทีละส่วน) + visualize/
├── 05_extract/<ชุด>/<engine>/<วิธี>/<extractor>/ ผลตาม schema (JSON) extractor = rules | llm | hybrid
├── etc/benchmarks/ · etc/compare/ · etc/experiments/   งานทดสอบ/เปรียบเทียบ
└── reports/                                     รายงานภาพ
```

ถ้ารันหนึ่งครั้งมีไฟล์อินพุตมากกว่า 2 ไฟล์ ผลของแต่ละไฟล์จะอยู่ในโฟลเดอร์ย่อยชื่อไฟล์ (เช่น `04_ocr/<ชุด>/paddle/page/<ไฟล์>/<ไฟล์>_p1.txt`) ไม่เกิน 2 ไฟล์ก็วางตรง ๆ (ตั้งที่ `NEST_OVER` ใน `src/fin_extractor/paths.py`)

เช่น `extract data/01_input/finance_doc_scan` → `data/05_extract/finance_doc_scan/paddle/page/rules/<ไฟล์>_p1.json` และข้อความดิบที่ `data/04_ocr/finance_doc_scan/paddle/page/`
`extract ... --reuse-raw` สกัดซ้ำจากข้อความดิบโดยไม่เรียก OCR/GPU ดู [docs/06-usage.md](docs/06-usage.md#โครงเก็บผลลัพธ์)

## โครงสร้างโฟลเดอร์

```
fin_extractor/
├── configs/default.yaml        ค่าตั้งต้นทั้งหมด (preprocess, engine, endpoint VLM/LLM, document_types)
├── assets/fonts/               ฟอนต์ Sarabun (SIL OFL) ใช้วาดชุดสังเคราะห์/ป้ายภาพ
├── data/                       01_input … 05_extract + etc/ + reports/ (ไม่ commit ยกเว้น .gitkeep)
├── docs/                       เอกสารอธิบายทีละขั้น
├── models/                     โมเดลที่ library โหลดเอง (paddlex/, OpenThai2.0-OCR-Markdown/) ไม่ commit
├── notebooks/preprocess_lab/   labkit.py + notebook เช็ค preprocess (สร้างด้วย make_preprocess_notebooks.py ไม่ commit .ipynb)
├── schemas/                    JSON Schema (Data Contract) — generate จาก schema.py
├── scripts/                    make_section_layouts / make_clean_sections / run_sections (ตัวห่อบางๆ ของ flow หลัก) · make_synthetic_samples + benchmark_preprocessing + benchmark_cer (ชุดสังเคราะห์) ·
│                               benchmark_real_scans + organize_preprocess_results + compare_modes + compare_file (สแกนจริง ไม่ใช้เฉลย) ·
│                               benchmark (ชุดมีเฉลย วัดความถูกต้องรายฟิลด์) · make_preprocess_notebooks · visualize_bbox ·
│                               oxr/ (OXR): extract_oxr / benchmark_oxr / visualize_oxr / explain_oxr
├── src/fin_extractor/
│   ├── schema.py               ★ Data Contract (pydantic)
│   ├── config.py
│   ├── pipeline.py             orchestrator
│   ├── cli.py                  python -m fin_extractor ... (preprocess · ocr · extract · sections · compare · schema)
│   ├── paths.py                ที่เดียวที่กำหนดว่าแต่ละขั้นเขียนข้อมูลที่ไหน (data/01_input … 05_extract)
│   ├── stages.py               preprocess ที่เก็บไว้ใช้ร่วมกัน + ข้อความดิบ
│   ├── doctypes.py             โปรไฟล์ชนิดเอกสาร (ฟิลด์/required/checks ต่อชนิด)
│   ├── runs.py                 manifest ของการรัน
│   ├── io/loader.py
│   ├── preprocessing/          หนึ่งไฟล์ต่อหนึ่งขั้น (แบบ learning/preprocess_finance) ลำดับตาม pipeline.py
│   │   ├── clean.py            ลบเส้นสกปรกจากสแกน (ก่อนหมุน) · denoise
│   │   ├── enhance.py          gray, upscale, CLAHE
│   │   ├── orient.py           ทิศทั้งหน้า 0/90/180/270 (Paddle doc-orientation)
│   │   ├── deskew.py           แก้เอียง (projection profile)
│   │   ├── crop.py             ตัดขอบ
│   │   ├── binarize.py         Otsu / adaptive / Sauvola
│   │   └── pipeline.py         รวมขั้นตอนตาม config: clean(streaks) → upscale → orient → deskew → crop → denoise → CLAHE → binarize
│   ├── ocr/                    base.py (interface กลาง) + paddle_engine.py
│   ├── sections/               ★ flow หลัก: layout.py (ผัง) · images.py (luma/flat/bin) · crops.py · build.py (ขั้น 2–3) · read.py (ขั้น 4 OCR รายส่วน + chunks.json) · flow.py (กติกาข้ามขั้น) · compare.py (คำสั่ง compare)
│   ├── vlm/                    base.py (VLMResult), client.py (Ollama/OpenAI-compatible), typhoon_ocr.py, openthai_ocr.py, oxr_local.py, prompts.py
│   ├── extraction/             patterns.py (regex), rule_extractor.py, profile_extractor.py, core_labels.py, llm_extractor.py, hybrid.py
│   └── validation/             tax_id.py, checks.py, scoring.py, evidence.py
└── tests/
```

## หมายเหตุการออกแบบ

- **Preprocess ชุดเดียว**: แก้เรขาคณิต (ตะแคง/เอียง) แต่คงสี ไม่ binarize ไม่ลด noise — ชุดสังเคราะห์วัดว่าภาพสีอ่านได้ครบที่สุดและเร็วที่สุด ส่วนสแกนจริง 18 ใบ (ไม่มีเฉลย) ภาพเทาไม่ต่างจากสี แต่ binarize (Otsu/Sauvola/adaptive) อ่านเลขภาษีได้น้อยลงและ NL-means ช้าลง 4 เท่าโดยไม่ช่วย (ดู [docs/02-preprocessing.md](docs/02-preprocessing.md#ผลวัดบนสแกนจริง))
- **ต่อคำภาษาไทย**: `OCRLine.text` เว้นวรรคตามระยะห่างจริงในภาพ (ไม่งั้นได้ `บ ร ิ ษั ท`) และแก้ `ํา` → `ำ`
- **Ollama ใช้ native API** เพราะ endpoint แบบ OpenAI ตั้ง `num_ctx` ไม่ได้ (default 4096 ทำให้ VLM ตอบไม่จบ)
- **LLM + rules + cross-check**: mode `vlm` รันทั้งสองแบบ เลือกผลที่ตัวเลขสอดคล้องกันมากกว่า แล้วเติม field ที่ขาดจากอีกฝั่ง
- **status**: `auto_approved` (field สำคัญครบ, ไม่มี check ไหน fail และผ่านอย่างน้อย 1 ข้อ) / `needs_review` / `failed` (หา total ไม่เจอ)
- **field_evidence**: ระดับหลักฐานรายฟิลด์ (`missing` / `not_applicable` / `conflict` / `verified` / `unverified`) จากกฎจริง/เท็จ ไม่มีน้ำหนักหรือ threshold ที่เดาเอง ไว้ไฮไลต์ในหน้าตรวจทานของนักบัญชี ทุกเอกสารยังต้องผ่านคนตรวจ
