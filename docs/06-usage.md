# ติดตั้งและวิธีใช้

## ติดตั้ง (เครื่องใหม่ตั้งแต่ clone)

ต้องมี Python 3.10 ขึ้นไป (ทดสอบกับ 3.13) PaddleOCR เป็น OCR engine ตัวเดียว ต้องลง `paddlepaddle` (ตัว framework) **ก่อน** แล้วค่อยลงโปรเจกต์

```powershell
git clone https://github.com/Pikcolo/fin-extractor.git
cd fin-extractor
python -m venv .venv
.venv\Scripts\activate                 # Windows (Linux/macOS: source .venv/bin/activate)

# 1) paddlepaddle เลือกอย่างใดอย่างหนึ่ง
pip install paddlepaddle-gpu==3.4.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu130/ --extra-index-url https://pypi.org/simple   # มี GPU
pip install paddlepaddle                                                                                                                  # ไม่มี GPU

# 2) โปรเจกต์ (paddleocr + vlm + pytest)
pip install -e ".[all]"

# 3) ตรวจ
python -c "import paddle; print(paddle.device.is_compiled_with_cuda())"   # ใช้ GPU ต้องได้ True
pytest                                                                     # ต้องผ่านทั้งหมด ไม่ต้องใช้ GPU/Ollama/ข้อมูลในโฟลเดอร์ data
```

- `paddlepaddle-gpu` ไม่อยู่บน PyPI ต้องดึงจาก index ของ Paddle และต้องมี `--extra-index-url`
  ไว้ดึงแพ็กเกจ `nvidia-*` (cuBLAS, cuDNN) จาก PyPI ไม่งั้นล้มที่ `nvidia-cudnn-cu13`
  เลือก `cuXXX` ให้ตรงกับ driver (ดู `nvidia-smi` → `CUDA Version`)
- โปรเจกต์ใช้ `opencv-contrib-python` ไม่ใช่ `headless` เพราะ `paddlex` (ที่ PaddleOCR ใช้)
  บังคับรุ่น contrib — สองแพ็กเกจเขียนลงโฟลเดอร์ `cv2` เดียวกัน ห้ามลงคู่กัน
- รันครั้งแรกจะโหลดโมเดลของ Paddle (ช้ากว่าปกติ) เก็บไว้ที่ `models/paddlex/official_models/` **ในโปรเจกต์** (ไม่ commit) ไม่ใช่ `~/.paddlex` โค้ดตั้ง `PADDLE_PDX_CACHE_HOME` ให้เอง (ถ้าตั้งไว้เองก่อนแล้วจะไม่ทับ) ย้ายเครื่องแล้วคัดลอกโฟลเดอร์ `models/` ไปด้วยก็ไม่ต้องโหลดใหม่ ตั้ง
  `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` เพื่อข้ามการเช็กเซิร์ฟเวอร์โมเดลทุกครั้งที่เริ่ม
  (PowerShell: `$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = "True"`)
- Windows: ใส่ `-X utf8` (`python -X utf8 -m fin_extractor ...`) ถ้าคอนโซลแสดงภาษาไทยเพี้ยนหรือ error `charmap`

### ส่วนเสริม (ตามที่จะใช้)

| สิ่งที่ต้องมี | ใช้กับ | ติดตั้ง |
|---|---|---|
| Ollama + Typhoon OCR | `--mode vlm` | `ollama pull scb10x/typhoon-ocr1.5-3b` |
| Ollama + LLM | การสกัดด้วย LLM (`--mode vlm`, `--llm on\|hybrid`) | `ollama pull qwen2.5:3b` (เปลี่ยนรุ่นที่ `llm_extractor.model`) |
| OpenThai 2.0 OCR (iApp, 0.63B) | `ocr --engine openthai`, `extract --mode vlm --engine openthai` | `pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128` แล้ว `pip install -e ".[openthai]"` และ `hf download iapp/OpenThai2.0-OCR-Markdown --local-dir models/OpenThai2.0-OCR-Markdown` (index cu128 มี torch สูงสุด 2.11; ถ้าลง torch จาก PyPI เฉย ๆ บน Windows จะได้รุ่น CPU) ทดสอบบน RTX 4050 6GB: VRAM สูงสุด ~2.4GB, 1 หน้า ~20 วินาที (รวมโหลดโมเดล) |
| ฟอนต์ | `scripts/make_synthetic_samples.py`, `scripts/oxr/visualize_oxr.py`, `scripts/visualize_bbox.py` | **มากับโปรเจกต์แล้ว** ที่ `assets/fonts/` (Sarabun, SIL OFL) ไม่ใช้ฟอนต์ของเครื่อง ไม่ต้องติดตั้ง |

**อะไรอยู่ในโปรเจกต์ อะไรอยู่นอก:**

| ของ | ที่อยู่ | หมายเหตุ |
|---|---|---|
| ฟอนต์ | `assets/fonts/` (commit) | ไม่ใช้ `C:/Windows/Fonts` |
| โมเดล Paddle (detection, recognition, orientation) | `models/paddlex/` (ไม่ commit) | ตั้งผ่าน `PADDLE_PDX_CACHE_HOME` อัตโนมัติ |
| โมเดล OpenThai 2.0 OCR | `models/OpenThai2.0-OCR-Markdown/` (ไม่ commit) | ดาวน์โหลดด้วย `hf download` ตามตารางส่วนเสริม |
| ไลบรารี Python | `.venv/` | |
| ข้อมูล ผลลัพธ์ cache | `data/` | |
| โมเดลของ Ollama (Typhoon, qwen) | **นอกโปรเจกต์** (ที่เก็บของ Ollama เอง เช่น `~/.ollama`) | Ollama เป็นบริการแยก ย้ายได้ด้วยตัวแปรแวดล้อม `OLLAMA_MODELS` ฝั่งเซิร์ฟเวอร์ของ Ollama โปรเจกต์ตั้งให้ไม่ได้ |

**ใช้ Typhoon API บน cloud แทน Ollama:** คัดลอก `.env.example` เป็น `.env` ใส่ `TYPHOON_API_KEY` แล้วตั้ง `vlm.provider: openai` และ
`vlm.base_url: https://api.opentyphoon.ai/v1` (ส่งเอกสารการเงินออกนอกเครื่อง ตรวจนโยบายข้อมูลก่อน) โหมด `ocr` ไม่ต้องใช้ทั้ง Ollama และ key

## ข้อมูลตัวอย่าง

ทุกโฟลเดอร์ใต้ `data/` อยู่ใน git เพียงแค่ `.gitkeep` เนื้อหาข้างในถูก `.gitignore` ไว้ (เอกสารจริงอาจมีข้อมูลภาษีและข้อมูลส่วนบุคคล)
หลัง clone จึงไม่มีเอกสารให้รัน มีสองทาง

1. **สแกนจริง:** รับชุดสแกนจากเจ้าของโปรเจกต์นอก git แล้ววางที่ `data/01_input/finance_doc_scan/` (ไฟล์ `.pdf`, `.jpg`, `.png`, `.tif`, `.bmp`, `.webp`)
2. **ชุดสังเคราะห์** (ไม่ต้องรอข้อมูลจริง): `python scripts/make_synthetic_samples.py` ได้ใบกำกับภาษีสมมติ 21 ใบพร้อมเฉลยที่ `data/01_input/synthetic/` (วาดด้วยฟอนต์ Sarabun ใน `assets/fonts/` ชุดที่สร้างไว้ก่อน 2026-11-11 ใช้ Tahoma ภาพจึงต่างจากชุดใหม่เล็กน้อย ตัวเลข CER/OXR บนชุดสังเคราะห์ในเอกสารมาจากชุด Tahoma เดิม)

## รันครั้งแรก

```powershell
python scripts/make_synthetic_samples.py                      # ถ้ายังไม่มีเอกสาร
python -m fin_extractor extract data/01_input/synthetic            # โหมด ocr (Paddle + rules) ไม่ต้องใช้ Ollama
```

ได้ผลที่ `data/05_extract/synthetic/paddle/page/rules/` (ข้อความดิบที่ `04_ocr/synthetic/paddle/page/` ดูโครงข้างล่าง) และบนหน้าจอจะแสดง status กับผล check ของแต่ละใบ
ตัวโหลดไฟล์กรองเฉพาะนามสกุลภาพ/PDF จึงไม่หยิบ `truth.json` ในโฟลเดอร์เดียวกัน

โหมด `vlm` (Typhoon + LLM) ต้องเปิด Ollama ก่อน และช้ากว่ามาก:

```powershell
python -m fin_extractor extract data/01_input/finance_doc_scan --mode vlm
```

## โครงเก็บผลลัพธ์

ข้อมูลทุกอย่างอยู่ใต้ `data/` และไม่ถูก commit (เก็บเฉพาะ `.gitkeep`) ตำแหน่งทั้งหมดกำหนดที่เดียวใน `src/fin_extractor/paths.py`
ชื่อไฟล์ผลของทุกขั้นเป็น `<ไฟล์>_p<หน้า>` เหมือนกัน จึงดูไฟล์เดียวข้ามขั้น/ข้าม engine ได้ และลบ engine ที่ไม่ใช้ได้ด้วยการลบโฟลเดอร์เดียว

```
data/
├── 01_input/<dataset>/                         ไฟล์อินพุต (dataset = ชื่อโฟลเดอร์ เช่น finance_doc_scan, synthetic)
├── 02_preprocess/<dataset>/<profile>/          ผล preprocess หนึ่งชุดตั้งค่า (profile ตั้งต้นชื่อ default)
│   ├── final/<ไฟล์>_p<หน้า>.png                ภาพสุดท้ายที่ส่งให้ OCR/VLM ทุกตัว ใช้ร่วมกัน (ไม่ preprocess ซ้ำ)
│   ├── steps/                                  ภาพทีละขั้น (เมื่อใส่ --steps)
│   └── settings.json · manifest.json
├── 03_sections/<dataset>/<profile>/            แบ่งเป็นส่วนจาก final/ (flow หลักของสแกนจริง ดู docs/13)
│   ├── layout/<ไฟล์>.json · <ไฟล์>_layout.png  ผังแบ่ง + ภาพกรอบสีไว้ดู/debug
│   └── images/<luma|flat|bin>/<ไฟล์>/sNN.png   ภาพรายส่วนที่เตรียมก่อน OCR + index.json
├── 04_ocr/<dataset>/<engine>/                  engine = paddle | typhoon | openthai | oxr
│   ├── page/<ไฟล์>_p<หน้า>.txt | .md           อ่านทั้งหน้า: ข้อความดิบ + .meta.json (+ manifest.json)
│   ├── <luma|flat|bin|color>/<ไฟล์>/           อ่านทีละ section: .md .sections.json .chunks.json (ที่มาของแต่ละท่อน)
│   └── visualize/                              ภาพให้ดูว่าอ่านอย่างไร
├── 05_extract/<dataset>/<engine>/<page|วิธี>/<extractor>/<ไฟล์>_p<หน้า>.json   ผลตาม schema (extractor = rules | llm | hybrid) + manifest.json
├── etc/                                        งานทดสอบ/เปรียบเทียบที่ข้าม engine/ข้ามไฟล์
│   ├── benchmarks/                             ผลวัด (CER, preprocess)
│   ├── compare/<เลขไฟล์>/                      เทียบรายไฟล์ + ค่าอ้างอิงที่คนอ่านจากภาพ
│   └── experiments/<ชื่อ>/                     การทดลองครั้งเดียวนอก pipeline
└── reports/                                    รายงานภาพ เช่น oxr_why/
```

**โฟลเดอร์ย่อยต่อไฟล์:** ถ้ารันหนึ่งครั้งมีไฟล์อินพุต **มากกว่า 2 ไฟล์** ผลของแต่ละไฟล์ (`02_preprocess/…/final/` และ `steps/`, `04_ocr/…/<วิธี>/`, `05_extract/…/<extractor>/`) จะอยู่ในโฟลเดอร์ย่อยชื่อไฟล์ เช่น `04_ocr/finance_doc_scan/paddle/page/BRWAC50DE3689DA_001312/BRWAC50DE3689DA_001312_p1.txt` ถ้าไม่เกิน 2 ไฟล์ (เช่นลองไฟล์เดียว) จะวางตรง ๆ ใต้โฟลเดอร์นั้น ค่า 2 อยู่ที่ `NEST_OVER` ใน `src/fin_extractor/paths.py` ตัวอ่าน (`--reuse-raw`, สคริปต์เทียบ/วาดรูป) หาได้ทั้งสองแบบ ผลที่ปนกันจึงอ่านได้ และ `manifest.json` ใช้พาธ `<ไฟล์>/<ไฟล์>_p1.json` ในช่อง `output`

| คำสั่ง | เขียนที่ | ข้างใน |
|---|---|---|
| `preprocess` | `data/02_preprocess/<dataset>/<profile>/` | `final/` ภาพสุดท้าย, `steps/` เมื่อใส่ `--steps`, `settings.json` (ค่าที่ทำให้ได้ภาพนี้), `manifest.json` |
| `ocr` | `data/04_ocr/<dataset>/<engine>/page/` | `<ไฟล์>_p<หน้า>.txt` (vlm เป็น `.md`) + `.meta.json` + `manifest.json` |
| `extract` | `data/05_extract/<dataset>/<engine>/<page\|วิธี>/<extractor>/` (ข้อความดิบที่ `data/04_ocr/<dataset>/<engine>/<page\|วิธี>/`) | `<ไฟล์>_p<หน้า>.json` ตาม `schemas/invoice.schema.json` + `manifest.json` |
| script วัดผล | `data/etc/benchmarks/` | ไฟล์ผลของแต่ละ script (ดูตารางข้างล่าง) |
| `sections` (หรือสคริปต์ `make_section_layouts.py` · `make_clean_sections.py`) | `data/03_sections/<dataset>/<profile>/` | `layout/<ไฟล์>.json` + `_layout.png` และ `images/<luma\|flat\|bin>/<ไฟล์>/` ภาพย่อยแยกโฟลเดอร์ตามวิธี (ขั้นอ่านเขียนที่ `04_ocr/<dataset>/<engine>/<วิธี>/` ขั้นสกัดที่ `05_extract/…`) |
| notebook เช็ค preprocess | `notebooks/preprocess_lab/<dataset>/` | สร้างด้วย `make_preprocess_notebooks.py` ไม่ commit เพราะมีภาพเอกสารจริง |
| การทดลอง | `data/etc/experiments/<ชื่อ>/` | การทดลองครั้งเดียวนอก pipeline |

- **engine** ในชื่อโฟลเดอร์: `--mode ocr` = `paddle`, `--mode vlm` = `typhoon` (หรือ `openthai` เมื่อใส่ `--engine openthai`) ส่วน **extractor**: `llm` เมื่อใช้ LLM สกัด, `hybrid` เมื่อ `--llm hybrid`, ไม่งั้น `rules`
- **OCR ครั้งเดียว สกัดได้หลายแบบ:** `extract` เก็บข้อความดิบไว้ที่ `04_ocr/<dataset>/<engine>/<วิธี>/` แล้ว `extract ... --reuse-raw` สกัดซ้ำจากข้อความนั้น (เปลี่ยน rules ↔ LLM, แก้ขั้นสกัดแล้วลองใหม่) **โดยไม่โหลด/เรียก OCR หรือ VLM และไม่ใช้ GPU**
- **ภาพ preprocess ใช้ร่วมกันทุก engine:** ถ้าตั้งค่า preprocess เปลี่ยนจากที่เก็บไว้ ระบบจะไม่ปนภาพคนละตั้งค่า ให้ใช้ `--profile <ชื่อใหม่>` หรือ `--force` (ล้างโฟลเดอร์ profile นั้นแล้วทำใหม่)
- รันซ้ำด้วยอินพุตและวิธีเดิมจะเขียนทับ ถ้าอยากแยกการทดลองใช้ `--out <โฟลเดอร์>` (ผลลงใต้ `<โฟลเดอร์>/04_ocr/<วิธี>/` และ `<โฟลเดอร์>/05_extract/<วิธี>/<extractor>/` แทน ภาพ preprocess ยังใช้ร่วมกัน)
- `manifest.json` เขียนตอนจบ ถ้ารันล้มกลางทางจะมี `"status": "failed"` พร้อมไฟล์ที่ทำเสร็จแล้ว

## คำสั่ง

| คำสั่ง | ขั้นที่ทำ | ผลลัพธ์ |
|---|---|---|
| `python -m fin_extractor preprocess <input> [--profile P] [--steps] [--force] [--orientation 90/180/270]` | 1 → 2 | `data/02_preprocess/<dataset>/<profile>/` |
| `python -m fin_extractor ocr <input> --engine paddle\|vlm\|openthai` | 1 → 3 | `data/04_ocr/<dataset>/<engine>/page/` |
| `python -m fin_extractor extract <input> --mode ocr\|vlm [--engine openthai] [--llm auto\|on\|off\|hybrid] [--reuse-raw [--variant luma\|flat\|bin\|color]] [--only 1309 …]` | 1 → 7 | `data/04_ocr/<dataset>/<engine>/<page|วิธี>/` และ `data/05_extract/<dataset>/<engine>/<page|วิธี>/<extractor>/` |
| `python -m fin_extractor sections <input> [--variant luma\|flat\|bin\|color] [--engine openthai\|oxr] [--from ขั้น] [--to ขั้น] [--only 1309 …] [--redo] [--table-mode whole\|cells\|bands\|rows] [--max-piece-frac F] [--fill-aware] [--drop-empty-rows] [--profile P]` | flow หลัก: preprocess → layout → images → ocr → extract (ชื่อขั้นใน `--from`/`--to`: `preprocess` `layout` `images` `ocr` `extract`) | `data/02_preprocess` … `data/05_extract` ตามขั้น |
| `python -m fin_extractor compare <input> [--engine openthai] [--variants luma flat bin] [--out ไฟล์.md]` | — | ตารางเทียบผลสกัดข้ามวิธี พิมพ์ที่หน้าจอ (ใส่ `--out` เขียน .md ด้วย) แถวที่เทียบไม่แฟร์ถูกติดธงและไม่นับในยอดรวม |
| `python -m fin_extractor schema` | — | `schemas/invoice.schema.json` (generate จาก `schema.py` ห้ามแก้ไฟล์ตรง ๆ) |
| `pytest` | รันเทสต์ | |

ตัวเลือกของ `extract`:

| ตัวเลือก | ความหมาย |
|---|---|
| `--mode ocr\|vlm` | `ocr` = Paddle + rules (ตั้งต้น), `vlm` = Typhoon OCR + LLM + rules |
| `--engine` | `--mode ocr`: `paddle` (ตั้งต้น) · `--mode vlm`: ค่าตั้งต้น = Typhoon, `openthai` = OpenThai 2.0 OCR |
| `--llm auto\|on\|off\|hybrid` | `auto` = ใช้ LLM เฉพาะโหมด vlm · `on` = rules และ LLM สกัดแยก เลือกชุดที่ check ดีกว่า · `hybrid` = rules นำ LLM เติมเฉพาะช่องที่ยังไม่เสร็จ (ผลลง `…/hybrid/`) |
| `--profile` | ชื่อชุดตั้งค่า preprocess (โฟลเดอร์ `02_preprocess/<dataset>/<profile>/` ตั้งต้น `default`) |
| `--reuse-raw` | สกัดซ้ำจากข้อความดิบที่เก็บไว้ใน `04_ocr/…` (ไม่โหลดหรือเรียก OCR/VLM ไม่ใช้ GPU) |
| `--variant luma\|flat\|bin\|color` | ใช้กับ `--reuse-raw` เท่านั้น: อ่านข้อความดิบของ flow section จาก `04_ocr/<dataset>/<engine>/<วิธี>/` และเขียนที่ `05_extract/…/<วิธี>/<extractor>/` (ไม่ใส่ = `page`) |
| `--only` | สกัดเฉพาะไฟล์ที่ชื่อลงท้ายด้วยเลขเหล่านี้ เช่น `--only 1309 1310` (ไม่ใส่ = ทุกไฟล์) |
| `--keep-raw` | เก็บข้อความดิบใน JSON ผลสุดท้ายด้วย (ตั้งต้นเก็บเฉพาะใน `04_ocr/…`) |
| `--out` | โฟลเดอร์ปลายทางเอง (แทน `data/04_ocr/…` และ `data/05_extract/…`) ผลอยู่ใต้ `<out>/04_ocr/<page\|วิธี>/` และ `<out>/05_extract/<page\|วิธี>/<extractor>/` |
| `--config` | YAML ที่ override `configs/default.yaml` (วางก่อนชื่อคำสั่ง) |
| `-v` | แสดง log (วางก่อนชื่อคำสั่ง เช่น `python -m fin_extractor -v extract ...`) |

## Script วัดผล

ไม่ใช่ส่วนของ pipeline ใช้วัดและเทียบ ผลทั้งหมดอยู่ใต้ `data/etc/benchmarks/` (ยกเว้นที่ระบุ)

| Script | ทำอะไร | ต้องมีมาก่อน | ผลลัพธ์ |
|---|---|---|---|
| `make_synthetic_samples.py` | สร้างใบกำกับภาษีสังเคราะห์ 21 ใบ พร้อมเฉลย | — | `data/01_input/synthetic/` (ภาพ + `truth.json`) |
| `benchmark_preprocessing.py` | วัดว่าแต่ละชุด preprocessing ทำให้ Paddle อ่านค่าสำคัญได้ครบกี่ค่าจาก 6 | `make_synthetic_samples.py` | `preprocessing_synthetic.json` |
| `benchmark_cer.py [--engines paddle vlm]` | วัด CER บนชุดสังเคราะห์ (ข้อความอ้างอิงดึงจาก `render_page` ไม่ต้องมีเฉลยเพิ่ม) `vlm` ต้องมี Ollama + Typhoon | `make_synthetic_samples.py` | `cer_synthetic_<engines>.json` |
| `benchmark_real_scans.py` | เทียบ preprocessing 13 ชุดบนสแกนจริง ไม่ใช้เฉลย (~10 นาที) | สแกนใน `data/01_input/finance_doc_scan/` | `real_scans.json` |
| `organize_preprocess_results.py` | จัดผลข้างบนเป็นโฟลเดอร์ต่อชุด | `benchmark_real_scans.py` | `preprocess_experiments/` |
| `compare_modes.py` | เทียบผล `extract` ของ `paddle/rules` กับ `typhoon/llm` รายใบ | `extract data/01_input/finance_doc_scan` ทั้ง `--mode ocr` และ `--mode vlm` (อ่าน `final/` ของสองชุดนั้น) | `compare_modes.md` |
| `make_section_layouts.py` → `make_clean_sections.py` → `run_sections.py` | สคริปต์เดี่ยวของคำสั่ง `sections` (ตัวห่อบางๆ ของ `src/fin_extractor/sections/`) รันทีละขั้น: ผังแบ่ง → ภาพ luma/flat/bin → อ่านทีละส่วน ดู [13](13-section-preprocess.md) ทุกตัวรับ `--config` ทดลองโดยไม่ทับผลจริง | ภาพ `final/` (ผ่าน `preprocess`) | `03_sections/…/layout/`, `03_sections/…/images/<วิธี>/<ไฟล์>/`, `04_ocr/<dataset>/<engine>/<วิธี>/<ไฟล์>/*.sections.json` + `.md` |
| `make_preprocess_notebooks.py <โฟลเดอร์ภาพ>` | สร้าง notebook เช็ค preprocess ไฟล์ละเล่ม (edge, threshold, แบ่ง section ตามตาราง) เครื่องมืออยู่ `notebooks/preprocess_lab/labkit.py` | สแกนใน `data/01_input/<dataset>/` และ `preprocess` (ภาพที่ตรงแล้ว) | `notebooks/preprocess_lab/<dataset>/*.ipynb` |
| `compare_file.py <เลขไฟล์>` | เทียบผลสกัดของไฟล์เดียวรายฟิลด์ ข้าม engine/extractor (ไม่ใช้ GPU) มีคอลัมน์อ้างอิงถ้าใส่ `reference.json` | ผล `extract` ของแต่ละ engine | `data/etc/compare/<ไฟล์>/` |
| `oxr/extract_oxr.py <input>` | OXR เป็นตัวอ่าน (ต้องเปิด container OXR ในเครื่อง ใช้ RAM/VRAM มาก) ต่อจากที่ค้างได้ | `preprocess` + container `oxr` | `data/04_ocr/<dataset>/oxr/page/` และ `data/05_extract/<dataset>/oxr/page/rules/` |
| `visualize_bbox.py <เลขไฟล์ ...>` · `--cached` | วาดกรอบ (bbox) ที่ Paddle ตรวจเจอลงบนภาพหลัง preprocess เป็นภาพ `words.jpg` (กรอบทุกคำ สีตามความมั่นใจ) และ `lines.jpg` (บรรทัดพร้อมเลขลำดับ) ไม่ทำ HTML Paddle ไม่เก็บกรอบไว้ในผล extract จึงต้องอ่านภาพใหม่เฉพาะไฟล์ที่ระบุ (หนัก ทำทีละไฟล์) แล้วเก็บ `ocr_words.json` ไว้วาดซ้ำ Typhoon ไม่มีกรอบ | `preprocess` (ภาพ final) | `data/04_ocr/<dataset>/paddle/visualize/<ไฟล์>/` |
| `oxr/visualize_oxr.py <เลขไฟล์>` · `oxr/explain_oxr.py` | รายงานภาพทีละขั้นของ OXR · รายงาน "ทำไม OXR ไม่แม่น" (ไม่ใช้ GPU) | ผล OXR ใน `04_ocr/<dataset>/oxr/` | `…/visualize/` · `data/reports/oxr_why/` |
| `oxr/benchmark_oxr.py` | วัด CER ภาพสังเคราะห์กับ free API ของ OXR (ห้ามใช้กับสแกนจริง) | `make_synthetic_samples.py` | `data/etc/benchmarks/oxr_synthetic_*.json` |
| `benchmark.py prepare --per-cell 2` | โหลดชุด Invoice Extraction Benchmark จาก Hugging Face (ต้องต่อเน็ต) พร้อมเฉลย; `--thai <โฟลเดอร์>` เพิ่มเอกสารไทยที่มีเฉลย `.json` คู่กัน | — | `data/01_input/benchmark/` |
| `benchmark.py run paddle\|paddle+llm\|vlm` | รันทีละ engine กับชุดข้างบน (`paddle+llm`, `vlm` ต้องมี Ollama) | `prepare` | `ieb/<engine>.jsonl` |
| `benchmark.py report` | สรุปความถูกต้องรายฟิลด์และเวลาของทุก engine ที่ `run` แล้ว | `run` | `ieb/report.md` |

ตัวอย่างแผนทดลองเทียบสองโหมดบนสแกนจริง:

```powershell
python -m fin_extractor extract data/01_input/finance_doc_scan --llm off     # → 04_ocr/finance_doc_scan/paddle/page/ และ 05_extract/finance_doc_scan/paddle/page/rules/
python -m fin_extractor extract data/01_input/finance_doc_scan --mode vlm    # → 04_ocr/finance_doc_scan/typhoon/page/ และ 05_extract/finance_doc_scan/typhoon/page/llm/ (OCR ครั้งเดียว)
python -m fin_extractor extract data/01_input/finance_doc_scan --mode vlm --llm off --reuse-raw   # rules จากข้อความ Typhoon เดิม ไม่ใช้ GPU
python -m fin_extractor extract data/01_input/finance_doc_scan --llm on --reuse-raw             # → 05_extract/finance_doc_scan/paddle/page/llm (LLM จากข้อความ Paddle เดิม ใช้แค่ Ollama)
python scripts/compare_modes.py                                          # → data/etc/benchmarks/compare_modes.md
```

## โครงสร้างโปรเจกต์

```
fin_extractor/
├── configs/default.yaml        ค่าตั้งต้นทั้งหมด
├── assets/fonts/               ฟอนต์ Sarabun (SIL OFL)
├── data/                       01_input/ (เอกสารต้นฉบับ) · 02_preprocess/ · 03_sections/ · 04_ocr/ · 05_extract/ · etc/{benchmarks,compare,experiments} · reports/ ไม่ถูก commit ยกเว้น .gitkeep
├── docs/                       เอกสารนี้
├── models/                     โมเดลที่ library โหลดเอง (ไม่ commit)
├── notebooks/preprocess_lab/   labkit.py + notebook เช็ค preprocess
├── schemas/invoice.schema.json Data Contract (generate จาก schema.py)
├── scripts/                    visualize_bbox (กรอบ Paddle) · benchmark (มีเฉลย) · benchmark_preprocessing + benchmark_cer + make_synthetic_samples (สังเคราะห์)
│                               · benchmark_real_scans + organize_preprocess_results + compare_modes + compare_file (สแกนจริง ไม่ใช้เฉลย)
│                               · make_section_layouts + make_clean_sections + run_sections (flow section) · make_preprocess_notebooks
│                               · oxr/ : extract_oxr + benchmark_oxr + visualize_oxr + explain_oxr (OXR)
├── src/fin_extractor/
│   ├── schema.py               Data Contract (pydantic)
│   ├── config.py
│   ├── pipeline.py             ตัวควบคุมลำดับ
│   ├── cli.py
│   ├── paths.py                ที่เดียวที่บอกว่าแต่ละขั้นเขียนข้อมูลที่ไหน
│   ├── stages.py               preprocess ที่เก็บไว้ใช้ร่วมกันทุก engine + ข้อความดิบ (04_ocr)
│   ├── doctypes.py             โปรไฟล์ชนิดเอกสาร (ฟิลด์/required/checks ต่อชนิด)
│   ├── runs.py                 manifest ของการรัน
│   ├── io/                     ขั้นที่ 1
│   ├── preprocessing/          ขั้นที่ 2
│   ├── ocr/                    ขั้นที่ 3A
│   ├── vlm/                    ขั้นที่ 3B (Typhoon, OpenThai, OXR) + client สำหรับ LLM
│   ├── sections/               flow หลัก: ผัง → ภาพรายส่วน → OCR รายส่วน (ดู docs/13) + compare
│   ├── extraction/             ขั้นที่ 4
│   └── validation/             ขั้นที่ 5–6 + evidence รายฟิลด์
└── tests/
```

## แก้ไฟล์แล้วต้องรันอะไร

| แก้อะไร | รันอะไร |
|---|---|
| โค้ดทั่วไป | `pytest` |
| `schema.py` | `pytest` + `python -m fin_extractor schema` (เพิ่ม `SCHEMA_VERSION` ถ้าโครงเปลี่ยน) |
| extraction / validation / prompt | `pytest` + ลอง `extract` กับเอกสารจริงอย่างน้อยหนึ่งใบ (prompt/LLM ต้องลองโหมด vlm ด้วย เทสต์ไม่ครอบพฤติกรรมของ LLM) |
| `configs/default.yaml` | ลอง `extract` ถ้าแก้ preprocessing หรือ `document_types` |
| ขั้นสกัด / `document_types` | `python -m fin_extractor extract <input> --reuse-raw` (เติม `--mode vlm` ถ้าเป็นข้อความ Typhoon) สกัดและตรวจซ้ำจากข้อความดิบใน `04_ocr/` ไม่เรียก OCR/GPU (วินาทีเดียว) เทียบก่อน/หลังบนข้อความเดิมได้ |
| `make_synthetic_samples.py` | ระวัง: ผลวัดใน [02](02-preprocessing.md) อ้างภาพชุดปัจจุบัน สร้างใหม่แล้วตัวเลขเดิมอาจเทียบไม่ได้ |
| docs | ไม่ต้องรัน แต่ตัวเลขผลวัดต้องตรงกับผลจริงล่าสุดใน `data/etc/benchmarks/` และ `data/05_extract/` |

## ข้อจำกัดที่ยังไม่ได้ทดสอบ

- **PaddleOCR** รันจริงแล้วบนเอกสาร 2 ใบ (RTX 4050, CUDA 13), ชุดสังเคราะห์ 21 ใบ และสแกนจริง 18 ใบ (`--mode ocr` ราว 2–5 วินาที/ใบ ไม่นับใบแรกที่โหลดโมเดล) ผลเทียบ engine ที่เคยลองดู
  อยู่ใน [03-text-reading.md](03-text-reading.md#ผลเทียบสาม-engine-บนเครื่องนี้) — ยังน้อยเกินกว่าจะสรุป
- **Prompt ของ Typhoon OCR** (`vlm/prompts.py`) อิงรูปแบบใน model card ควรเทียบกับ model card ของรุ่นที่ใช้อีกครั้ง
- **Typhoon API บน cloud**: มี config แล้วแต่ยังไม่ได้เรียกจริง
- **rules ถูกปรับให้เข้ากับ `BITKUB-001.jpg`** ชุดสแกนจริง 18 ใบ (`data/01_input/finance_doc_scan`) ไม่มีใบไหนได้ `auto_approved` อย่างถูกต้อง
  เพราะมีเอกสารหลายชนิดนอกใบกำกับภาษี (หนังสือรับรอง 50 ทวิ, ใบสำคัญรับเงินลายมือ, ใบเสร็จศุลกากร) และ rules บางส่วนพลาดแม้ OCR อ่านครบ (เช่น 001310)
  ตัวเลขสถานะล่าสุดอยู่ใน [README ของ docs](README.md) ดูเหตุผลที่อนุมัติอัตโนมัติไม่ได้ใน [05](05-validation-output.md#ขั้นที่-6-สถานะเอกสาร) และผล preprocess ใน [02-preprocessing.md](02-preprocessing.md#ผลวัดบนสแกนจริง)
- **LLM ขนาด 3B** ยังสลับ field ยอดเงินได้ ใช้รุ่นใหญ่ขึ้นจะเสถียรกว่า
- **วันที่:** ถ้าช่องวันที่ถูกบังหรือว่าง ระบบจะหยิบวันที่แรกในเอกสาร ซึ่งอาจไม่ใช่วันที่ออกเอกสาร
