"""
src/api_answer_generator.py  (LUỒNG RIÊNG cho app demo — KHÔNG dùng trong main.py / fast_retrieval.py)

Sinh câu trả lời qua API hosted OpenAI-compatible (Together / DeepInfra / Novita...) thay vì nạp
Qwen-7B 4-bit tại chỗ — vì máy demo chỉ có CPU. Tái sử dụng NGUYÊN VẸN system prompt và
build_user_prompt() của bản thi (src/answer_generator.py) để giữ chất lượng/định dạng câu trả lời
giống hệt đường thi đấu; chỉ đổi backend suy luận và bật streaming.

Hậu xử lý giống bản gốc: bóc <think>...</think>, gộp mọi khoảng trắng thành 1 đoạn văn, hoa chữ đầu.
"""
import re
import logging
from typing import Iterator, List, Dict

from config.settings import Settings
from src.answer_generator import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
_THINK_OPEN = re.compile(r"<think>.*", flags=re.DOTALL)


def clean_answer(text: str) -> str:
    """Hậu xử lý GIỐNG src/answer_generator.py:245-253 — 1 đoạn văn, không thẻ suy nghĩ, hoa chữ đầu."""
    if not text:
        return ""
    text = _THINK_BLOCK.sub("", text)
    text = _THINK_OPEN.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text:
        text = text[0].upper() + text[1:]
    return text


class ApiAnswerGenerator:
    """
    Bộ sinh câu trả lời qua API. Không load model nặng → khởi tạo tức thì.
    Dùng client `openai` (tương thích endpoint OpenAI-compatible qua base_url).
    """

    def __init__(self):
        from openai import OpenAI  # import trễ để không bắt buộc khi chạy đường thi

        if not Settings.LLM_API_KEY:
            logger.warning(
                "[ApiLLM] LLM_API_KEY rỗng — hãy đặt trong .env. "
                "App sẽ lỗi khi gọi sinh câu trả lời."
            )
        self.client = OpenAI(
            base_url=Settings.LLM_API_BASE_URL,
            api_key=Settings.LLM_API_KEY or "sk-missing",
        )
        self.model = Settings.LLM_API_MODEL
        logger.info(f"[ApiLLM] Sẵn sàng: model={self.model} @ {Settings.LLM_API_BASE_URL}")

    def _messages(self, query: str, contexts: List[Dict]) -> List[Dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(query, contexts)},
        ]

    def stream_raw(
        self,
        query: str,
        contexts: List[Dict],
        temperature: float = Settings.LLM_TEMPERATURE,
    ) -> Iterator[str]:
        """
        Stream các mảnh (delta) text THÔ từ API theo thời gian thực (để UI hiển thị dần).
        Lưu ý: đây là text thô — thẻ <think> (nếu model sinh) chỉ được lọc trọn ở clean_answer().
        """
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self._messages(query, contexts),
            temperature=temperature,
            top_p=Settings.LLM_TOP_P,
            max_tokens=max(1024, getattr(Settings, "LLM_MAX_NEW_TOKENS", 1024)),
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                yield piece

    def generate(
        self,
        query: str,
        contexts: List[Dict],
        temperature: float = Settings.LLM_TEMPERATURE,
    ) -> str:
        """Sinh trọn câu trả lời (không stream) + hậu xử lý. Dùng khi cần bản gọn cuối cùng."""
        buffer = "".join(self.stream_raw(query, contexts, temperature))
        return clean_answer(buffer)
