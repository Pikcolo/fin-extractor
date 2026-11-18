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
          └─ สายงานล่าสุด: แบ่งหน้าเป็น section → OpenThai OCR ทีละส่วน → Markdown ต่อกัน → rules   (ดู "สายงานล่าสุด")
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

เอกสารจริงไม่อยู่ใน git (ทุกอย่างใต้ `data/` ถูก `.gitignore`) วางไฟล์ของคุณที่ `data/raw/` หรือสร้างชุดสังเคราะห์:

```powershell
python scripts/make_synthetic_samples.py                  # ใบกำกับภาษีสมมติ 21 ใบ → data/raw/synthetic/
python -X utf8 -m fin_extractor extract data/raw/synthetic   # Paddle + rules → data/result_ocr/paddle/synthetic/final/rules/
```

### คำสั่งหลัก

```powershell
python -m fin_extractor preprocess data/raw --steps           # ดูผล preprocess ทีละขั้น
python -m fin_extractor ocr data/raw/BITKUB-001.jpg           # ข้อความดิบ (--engine vlm = Typhoon OCR)
python -m fin_extractor extract data/raw                      # pipeline เต็ม → JSON (ค่าตั้งต้น --mode ocr)
python -m fin_extractor extract data/raw --mode vlm           # Typhoon OCR + LLM + rules (ต้องเปิด Ollama)
python -m fin_extractor schema                                # export JSON Schema (หลังแก้ schema.py)
pytest
```

โหมด `vlm` ต้อง `ollama pull scb10x/typhoon-ocr1.5-3b` และ `ollama pull qwen2.5:3b` (หรือใช้ Typhoon API ตั้งค่าใน `.env`)

### สายงานล่าสุด: แบ่ง section แล้วอ่านด้วย OpenThai (สแกนจริง)

แบ่งหน้าเป็นแถบ/ท่อน (ตารางเป็นก้อนเดียว ตัดท่อนสูงไม่เกิน 0.3 ของหน้า ไม่ใช้ OCR คัดส่วน) เตรียมภาพรายส่วน แล้วให้ OpenThai อ่านทีละส่วน
ข้อมูลทั้งหมดอยู่ใต้ `data/` ใช้ GPU เฉพาะขั้นที่ 4 ([docs/14](docs/14-section-preprocess.md))

```powershell
python -m fin_extractor preprocess data/raw/finance_doc_scan                                   # 1 ภาพทั้งหน้า ตรงแล้ว (สี)
python -X utf8 scripts/make_section_layouts.py data/raw/finance_doc_scan                       # 2 ผังแบ่ง section
python -X utf8 scripts/make_clean_sections.py data/raw/finance_doc_scan                        # 3 ภาพรายส่วน clean/<luma|flat|bin>/<ไฟล์>/sNN.png (CPU)
python -X utf8 scripts/run_sections.py data/raw/finance_doc_scan --engine openthai --variant luma   # 4 อ่านทีละส่วน (GPU)
python -m fin_extractor extract data/raw/finance_doc_scan --mode vlm --engine openthai --llm off --reuse-raw --variant luma   # 5 สกัดด้วย rules (CPU)
```

- **ค่าตั้งต้นคือ `luma`**: เทียบ 3 แบบบน 18 ใบ (ไม่มีเฉลย วัดแค่ความครบ) luma ขาดฟิลด์รวม 14, flat 16, bin (ตัวขาวบนดำ) 28 ([docs/14](docs/14-section-preprocess.md))
- `--variant flat|bin` ผลแยกโฟลเดอร์ (`raw_data_<วิธี>/`, `final_<วิธี>/`) ไม่ทับกัน
- ข้อมูลภาษีจริงไม่ถูก commit และผลวัดทั้งหมดไม่ใช่คะแนนความถูกต้อง
- ข้อควรระวัง: ตรวจ `orientation_error` ใน `manifest.json` ของ preprocess ก่อนเชื่อผล (ตัวตรวจทิศอาจล้มเงียบ)

### ผลลัพธ์อยู่ที่ไหน

ข้อมูลทั้งหมดอยู่ใต้ `data/` (ไม่ commit) แยกตามขั้น แล้วแยกตาม engine:

```
data/
├── raw/<dataset>/                       ไฟล์อินพุต
├── preprocess/<dataset>/<profile>/      final/ ภาพสุดท้ายที่ทุก engine ใช้ร่วมกัน · steps/ ภาพทีละขั้น · sections/ ผังและภาพรายส่วน (features/ clean/<วิธี>/)
├── result_ocr/<engine>/<dataset>/       engine = paddle | typhoon | oxr | openthai
│   ├── raw_data/                        ข้อความดิบจาก OCR (เขียนครั้งเดียวต่อ engine)
│   ├── final/<rules|llm>/               ผลตาม schema (JSON)
│   └── visualize/                       ภาพให้ดูว่าอ่านอย่างไร
│   (openthai: raw_data_<luma|flat|bin>/ ข้อความดิบรายส่วน · final_<วิธี>/rules/ ผลตาม schema)
├── etc/benchmarks/ · etc/compare/       งานทดสอบ/เปรียบเทียบ (ผลวัด / เทียบรายไฟล์)
└── reports/                             รายงานภาพ
```

ถ้ารันหนึ่งครั้งมีไฟล์อินพุตมากกว่า 2 ไฟล์ ผลของแต่ละไฟล์จะอยู่ในโฟลเดอร์ย่อยชื่อไฟล์ (เช่น `raw_data/<ไฟล์>/<ไฟล์>_p1.txt`) ไม่เกิน 2 ไฟล์ก็วางตรง ๆ (ตั้งที่ `NEST_OVER` ใน `src/fin_extractor/paths.py`)

เช่น `extract data/raw/finance_doc_scan` → `data/result_ocr/paddle/finance_doc_scan/final/rules/<ไฟล์>_p1.json` และข้อความดิบที่ `raw_data/`
`extract ... --reuse-raw` สกัดซ้ำจากข้อความดิบโดยไม่เรียก OCR/GPU ดู [docs/06-usage.md](docs/06-usage.md#โครงเก็บผลลัพธ์)

## โครงสร้างโฟลเดอร์

```
fin_extractor/
├── configs/default.yaml        ค่าตั้งต้นทั้งหมด (preprocess, engine, endpoint VLM/LLM, document_types)
├── data/                       raw/ เอกสารต้นฉบับ · output/ ผล (ไม่ commit ยกเว้น .gitkeep)
├── schemas/                    JSON Schema (Data Contract) — generate จาก schema.py
├── scripts/                    make_section_layouts → make_clean_sections → run_sections (สายงานล่าสุด) · make_synthetic_samples + benchmark_preprocessing + benchmark_cer (ชุดสังเคราะห์) ·
│                               benchmark_real_scans + organize_preprocess_results + compare_modes (สแกนจริง ไม่ใช้เฉลย) ·
│                               benchmark (ชุดมีเฉลย วัดความถูกต้องรายฟิลด์)
├── src/fin_extractor/
│   ├── schema.py               ★ Data Contract (pydantic)
│   ├── config.py
│   ├── pipeline.py             orchestrator
│   ├── cli.py                  python -m fin_extractor ...
│   ├── runs.py                 โครงเก็บผล + manifest
│   ├── io/loader.py
│   ├── preprocessing/
│   │   ├── enhance.py          gray, upscale, denoise, CLAHE
│   │   ├── geometry.py         orientation (Paddle doc-orientation), deskew (projection profile), crop
│   │   ├── binarize.py         Otsu / adaptive / Sauvola
│   │   └── pipeline.py         รวมขั้นตอนตาม config
│   ├── ocr/                    base.py (interface กลาง) + paddle_engine.py
│   ├── vlm/                    client.py (Ollama/OpenAI-compatible), typhoon_ocr.py, openthai_ocr.py, oxr_local.py, prompts.py
│   ├── routing/                relevance.py (คัดภาพย่อยด้วยป้าย/ค่า ยังไม่อยู่ในสายงานล่าสุด)
│   ├── extraction/             patterns.py (regex), rule_extractor.py, llm_extractor.py, hybrid.py
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
