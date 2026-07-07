"""
app/main.py

Backend FastAPI cho app demo Legal-RAG. Chạy trên máy GPU cloud (đã load embedding + reranker),
gọi LLM qua API + Qdrant Cloud. Expose ra ngoài bằng tunnel (cloudflared/ngrok).

Endpoints:
  GET  /            → trang demo (app/static/index.html)
  GET  /health      → trạng thái service
  POST /ask         → SSE stream: event `citations` (ngay) → nhiều event `token` → `done`/`error`

Chạy:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
import os
import json
import logging
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.service import RagService
from src.api_answer_generator import clean_answer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app.main")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI(title="Legal-RAG Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo: cho phép mọi origin (qua tunnel)
    allow_methods=["*"],
    allow_headers=["*"],
)

# Nạp service MỘT LẦN lúc khởi động (không dùng biến toàn cục lười để lỗi sớm nếu thiếu cấu hình).
_service: Optional[RagService] = None


@app.on_event("startup")
def _startup():
    global _service
    _service = RagService()


class AskRequest(BaseModel):
    question: str


def _sse(event: str, data) -> str:
    """Đóng gói một sự kiện SSE."""
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


@app.get("/health")
def health():
    return {"status": "ok", "ready": _service is not None}


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/ask")
def ask(req: AskRequest):
    question = (req.question or "").strip()
    if not question:
        return JSONResponse({"error": "Câu hỏi trống."}, status_code=400)
    if _service is None:
        return JSONResponse({"error": "Service chưa sẵn sàng."}, status_code=503)

    def event_stream():
        try:
            # --- Giai đoạn nhanh: retrieval + rerank → gửi citations NGAY ---
            ranked = _service.retrieve_and_rerank(question)
            citations, rel_docs, rel_articles = _service.build_citations(ranked)
            yield _sse("citations", {
                "citations": citations,
                "relevant_docs": rel_docs,
                "relevant_articles": rel_articles,
            })

            if not ranked:
                yield _sse("token", "Xin lỗi, tôi không tìm thấy văn bản pháp luật liên quan đến câu hỏi này.")
                yield _sse("done", {"answer": ""})
                return

            # --- Giai đoạn chậm: stream câu trả lời từ LLM API ---
            buffer = []
            for piece in _service.stream_answer(question, ranked):
                buffer.append(piece)
                yield _sse("token", piece)

            yield _sse("done", {"answer": clean_answer("".join(buffer))})
        except Exception as e:  # noqa: BLE001
            logger.exception("Lỗi khi xử lý /ask")
            yield _sse("error", {"message": f"Lỗi hệ thống: {e}"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Mount static (CSS/ảnh nếu có) — đặt sau các route để không nuốt "/".
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
