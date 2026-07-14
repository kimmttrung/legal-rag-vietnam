# NextGen — Trợ lý Pháp luật AI cho Doanh nghiệp nhỏ và vừa

Hệ thống **RAG (Retrieval-Augmented Generation)** hỏi–đáp pháp luật Việt Nam (trọng tâm **SME**):
mỗi câu hỏi được **truy hồi** điều luật liên quan → **rerank** → **LLM sinh câu trả lời** **luôn kèm
trích dẫn văn bản gốc kiểm chứng được**. Dự án gồm **2 sản phẩm**:

- 🌐 **Web app trực tiếp** (`app/` + `web/`) — chat streaming, có đăng nhập & lưu lịch sử, đã online 24/7.
- 🏆 **Pipeline thi đấu** (`fast_retrieval.py`) — chạy 2000 câu, tối ưu điểm **F2** truy hồi, đóng gói `submission.zip`.

---

## 🚀 Trải nghiệm ngay

| | |
|---|---|
| 🌐 **Demo online** | **https://maitrung21-nextgen-legal-rag.hf.space** |
| ▶️ **Video demo** | https://www.youtube.com/watch?v=VVD1SANn7RE |
| 📱 **Quét QR** | (ảnh QR bên dưới) |

> ⏳ **Lưu ý khi mở demo:** app chạy trên **Hugging Face Spaces bản miễn phí** nên sẽ **"ngủ" khi lâu không có người dùng**.
> Lần truy cập đầu tiên có thể mất **~30–60 giây (đôi khi vài phút)** để hệ thống "thức dậy" và nạp mô hình.
> Nếu thấy **tải lâu / trắng trang / báo Starting**, đây **KHÔNG phải lỗi** — cứ **chờ rồi tải lại (F5)**.
> Sau khi đã chạy, các lần vào sau sẽ nhanh (cho tới khi Space ngủ lại).


<p align="center">
  <img src="https://raw.githubusercontent.com/kimmttrung/legal-rag-vietnam/main/docs/qr_nextgen.png" width="180" alt="QR mở demo NextGen">
  <br><em>Quét để mở demo trên điện thoại</em>
</p>

---

## ✨ Tính năng web app

- **Hỏi–đáp có trích dẫn:** câu trả lời tiếng Việt kèm panel **"Văn bản liên quan"** (số hiệu · Điều · đoạn trích · **link tới văn bản gốc**).
- **Streaming:** trích dẫn hiện ngay (~2s), câu trả lời chạy dần từng chữ.
- **Đăng nhập Google + Lịch sử hội thoại** (Supabase Auth + Postgres, bảo vệ bằng RLS).
- **Responsive** desktop/mobile; chip gợi ý câu hỏi; nút sao chép.
- **Tuân thủ quy định BTC:** LLM **< 14 tỷ tham số**.

---

## 🧠 Cách hoạt động

### A) Web app (RAG thời gian thực)
```
Câu hỏi ─► Embedding ─► Truy hồi lai (Qdrant dense + BM25 sparse → RRF) ─► Rerank
        ─► [gửi CITATIONS ngay] ─► LLM < 14B qua API (streaming) ─► [gửi ANSWER dần]
        ─► (nếu đăng nhập) lưu hội thoại vào Supabase
```
Backend **FastAPI** phục vụ luôn React build (gộp 1 nơi), gọi **Qdrant Cloud** + **LLM API**
(OpenAI-compatible, model < 14B). Chi tiết chạy local & deploy: mục **Triển khai** bên dưới.

### B) Pipeline thi đấu (tối ưu F2)
```
Câu hỏi ──(3 biến thể Multi-Query)
   ├─(1) Hybrid Retrieval ── Dense (Qdrant + Vietnamese_Embedding) + Sparse (BM25 + underthesea) ─► RRF → TOP-30
   ├─(2) Rerank ─────────── BAAI/bge-reranker-large → pool TOP-8 (--pool-k)
   ├─(3) LLM sinh answer ── Qwen/Qwen2.5-7B-Instruct (4-bit) → câu trả lời tiếng Việt
   ├─(4) Chọn trích dẫn ─── HỢP (∪): top-2 rerank ∪ {Điều LLM trích ∩ pool} → tăng recall, vẫn grounded
   └─(5) Đóng gói ────────── results.json → submission.zip (đúng định dạng BTC)
```
Chi tiết mô hình & checkpoint: [`docs/MODELS.md`](docs/MODELS.md).

### 📊 Kết quả (tập đánh giá của đội)
| Chỉ số | Điều (Articles) | Văn bản (Docs) |
|---|---|---|
| **F2 (macro)** | **0.67** | **0.73** |
| Recall | 0.71 | 0.79 |
| Precision | 0.62 | 0.62 |

---

## 🏗️ Ngăn xếp công nghệ

| Thành phần | Công nghệ |
|---|---|
| Embedding | `AITeamVN/Vietnamese_Embedding` (bge-m3, 1024-dim) |
| Vector DB | Qdrant Cloud — collection `law_2026` (~30k đoạn) |
| Sparse retrieval | BM25 + `underthesea` |
| Reranker | `BAAI/bge-reranker-large` |
| LLM (thi đấu) | `Qwen/Qwen2.5-7B-Instruct` 4-bit — GPU |
| LLM (web app) | Model **< 14B** qua API OpenAI-compatible (vd Groq `llama-3.1-8b-instant`) |
| Auth + Lịch sử | Supabase (Google OAuth + Postgres/RLS) |
| Backend / Frontend | FastAPI (SSE) + React (Vite) |
| Hạ tầng demo | Hugging Face Spaces (Docker, CPU free) |

---

## 📦 Triển khai web app (Hugging Face Spaces — CPU free)

Web app chạy trên **HF Spaces (16GB RAM, free)**: FastAPI phục vụ cả API + React build. Hướng dẫn đầy
đủ: [`docs/DEPLOY_HF_SPACES.md`](docs/DEPLOY_HF_SPACES.md). Tóm tắt:

**1) Biến môi trường cần đặt** (Space → Settings → Variables and secrets):
```
QDRANT_URL, QDRANT_API_KEY                    # Qdrant Cloud
LLM_API_BASE_URL, LLM_API_KEY, LLM_API_MODEL  # LLM < 14B qua API (vd Groq)
SUPABASE_URL, SUPABASE_ANON_KEY               # đăng nhập Google + lịch sử
```

**2) Đẩy code lên Space** (nhánh `deploy` = 1 commit sạch, LẶP LẠI mỗi lần cập nhật):
```bash
git branch -D deploy
git checkout --orphan deploy
git rm -r --cached docs/images
git rm --cached fast-retrieval.ipynb pipeline-intersect.ipynb piplne-and-debug.ipynb
git commit -m "Deploy NextGen Legal RAG (HF Spaces)"
git push hf deploy:main --force
git checkout -f main
```
> Vì lịch sử `main` chứa corpus lớn (LFS) + ảnh/notebook → HF chặn nếu push thẳng `main`.
> Nhánh `deploy` là orphan 1 commit nên HF chấp nhận. Docker tự build React + tải model + dựng BM25.

**Chạy thử local:**
```bash
pip install -r requirements-app.txt      # deps gọn cho web app
cd web && npm install && npm run build && cd ..   # build React → app/static
uvicorn app.main:app --host 127.0.0.1 --port 8000 # mở http://127.0.0.1:8000
```

---

## 🧩 Chạy pipeline thi đấu (tái hiện kết quả nộp)

### Yêu cầu môi trường
| Thành phần | Yêu cầu |
|---|---|
| Python | 3.10 – 3.11 |
| GPU | ≥ 16GB VRAM (Kaggle T4/P100) — để nạp LLM 4-bit |
| Đĩa | ~20GB (cache checkpoint HuggingFace) |
| Mạng | Internet (tải checkpoint + Qdrant Cloud) |

### Cách A — Kaggle (đúng môi trường đã nộp, khuyến nghị)
Mở [`pipeline-intersect.ipynb`](pipeline-intersect.ipynb) trên Kaggle (Accelerator = GPU T4×2/P100).

1. **Bật GPU:** Settings → Accelerator → **GPU T4 ×2**.
2. **Secrets Qdrant:** Add-ons → Secrets → thêm `QDRANT_URL`, `QDRANT_API_KEY`.
3. **Upload dataset câu hỏi:** Add Input → Upload → kéo `R2AIStage1DATA.json` (muốn test N câu thì cắt file còn N câu rồi upload).
4. **Dán đường dẫn** file vào biến `INPUT_FILE` ở **cell số 5**.
5. **Run All** → kết quả `/kaggle/working/results.json` + `submission.zip`.

> Ảnh hướng dẫn từng bước (mở trên GitHub):
> [GPU](https://raw.githubusercontent.com/kimmttrung/legal-rag-vietnam/main/docs/images/01_kaggle_gpu.png) ·
> [Secrets](https://raw.githubusercontent.com/kimmttrung/legal-rag-vietnam/main/docs/images/02_kaggle_secrets.png) ·
> [Upload](https://raw.githubusercontent.com/kimmttrung/legal-rag-vietnam/main/docs/images/03_kaggle_upload_dataset.png) ·
> [Input path](https://raw.githubusercontent.com/kimmttrung/legal-rag-vietnam/main/docs/images/04_kaggle_input_path.png) ·
> [Kết quả](https://raw.githubusercontent.com/kimmttrung/legal-rag-vietnam/main/docs/images/05_kaggle_output.png)

### Cách B — Dòng lệnh (local/server có GPU)
```bash
# Cài đặt
git clone https://github.com/kimmttrung/legal-rag-vietnam.git && cd legal-rag-vietnam
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
# tạo .env với QDRANT_URL, QDRANT_API_KEY

# Chạy đầy đủ 2000 câu → tạo bài nộp
python fast_retrieval.py --input data/R2AIStage1DATA.json \
    --output output/results.json --llm-answer --pool-k 8 --max-select 5

# Chạy thử nhanh 50 câu
python fast_retrieval.py --input data/R2AIStage1DATA.json \
    --output output/results_50.json --llm-answer --pool-k 8 --max-select 5 --num-questions 50

# Đóng gói
cd output && zip submission.zip results.json
```

Mỗi bản ghi `results.json`:
```json
{
  "id": "1",
  "question": "...",
  "answer": "Theo quy định tại Điều ...",
  "relevant_docs": ["67/2014/QH13|Luật Đầu tư"],
  "relevant_articles": ["67/2014/QH13|Luật Đầu tư|Điều 5"]
}
```

---

## 🗂️ Dữ liệu & Mô hình

| File | Mô tả |
|---|---|
| `data/law_corpus_clean.json` | ~30k đoạn văn bản pháp lý đã chunk (corpus BM25 + nguồn rerank) — LFS |
| `data/law_manifest.json` | Map số hiệu → metadata chuẩn (self-verify + chuẩn hóa output) |
| `data/doc_url_map.json` | Map số hiệu → URL văn bản gốc (dùng cho link trích dẫn ở web app) |
| `data/bm25_corpus.pkl` | Index BM25 (tự sinh từ corpus nếu chưa có) |
| `data/R2AIStage1DATA.json` | Bộ 2000 câu hỏi thi |

- **3 checkpoint pretrained công khai** (KHÔNG fine-tune) + **1 vector index tự tạo** (`law_2026`). Phiên bản pin chính xác: [`docs/MODELS.md`](docs/MODELS.md).
- **Dữ liệu tự xây:** crawl ~612 văn bản từ Thư viện Pháp luật → bóc tách 4 cấp (Chương→Điều→Khoản→Điểm) → QA → chunk theo Điều → embed → Qdrant.
- **Link chia sẻ dữ liệu/snapshot (Drive):** [NEXTGEN-legal-rag-data](https://drive.google.com/drive/folders/1mDoUfFrl8k3HNN6xlKsZpGdklU_A0Ifk?usp=sharing)

---

## 📁 Cấu trúc thư mục

```
legal-rag-vietnam/
├── app/                     # WEB APP — FastAPI backend (SSE) + React build (app/static)
│   ├── main.py              #   endpoints: /ask (SSE), /config, phục vụ static
│   └── service.py           #   RagService: retrieve → rerank → citations → stream
├── web/                     # WEB APP — mã nguồn React (Vite): chat, sidebar, đăng nhập, lịch sử
├── fast_retrieval.py        # PIPELINE THI — INTERSECT/HỢP (điểm chạy chính khi nộp)
├── main.py                  # PIPELINE THI — bản đầy đủ (retrieve→rerank→generate→self-verify→package)
├── src/                     # Module dùng chung
│   ├── hybrid_retriever.py  #   Hybrid retrieval (dense + sparse + RRF)
│   ├── reranker.py          #   Cross-encoder reranker
│   ├── reference_extractor.py#  Trích relevant_docs/articles (top-N rerank)
│   ├── answer_generator.py  #   LLM 4-bit sinh answer (thi đấu)
│   ├── answer_intersect.py  #   Chọn trích dẫn (giao/hợp answer ∩ pool)
│   ├── api_answer_generator.py# LLM qua API + streaming (web app)
│   └── ...                  #   index_bm25, self_verifier, post_processor, evaluator...
├── config/settings.py       # Toàn bộ tham số + model ID + biến môi trường
├── scripts/build_url_map.py # Sinh data/doc_url_map.json từ CSV số hiệu+URL
├── Dockerfile               # Build web app cho HF Spaces (React + FastAPI)
├── data/                    # Corpus, manifest, url map, bộ câu hỏi
└── docs/                    # MODELS.md, DEPLOY_HF_SPACES.md, images/, qr_nextgen.png
```

---

## 🛠️ Xử lý sự cố

| Triệu chứng | Cách xử lý |
|---|---|
| Web: `/config` trả rỗng, không đăng nhập được | Chưa đặt `SUPABASE_URL`/`SUPABASE_ANON_KEY` trong `.env`/Space → thêm rồi restart |
| Web: câu trả lời lỗi (citations vẫn hiện) | Sai `LLM_API_*` (key/model/endpoint). Model phải **< 14B** và còn hạn mức |
| Deploy HF bị chặn "files larger than 10 MiB / binary" | Dùng nhánh `deploy` orphan (đã gỡ corpus-history + ảnh) — xem mục Triển khai |
| Thi: `CUDA out of memory` khi nạp LLM | Cần GPU ≥16GB, nạp 4-bit; giảm `MAX_CONTEXT_CHARS` trong `config/settings.py` |
| Kết nối Qdrant lỗi/rỗng | Kiểm tra `QDRANT_URL`/`QDRANT_API_KEY`; xác nhận collection `law_2026` tồn tại |
| BM25 index không có | Tự sinh từ corpus lần chạy đầu, hoặc `python -m src.index_bm25` |
