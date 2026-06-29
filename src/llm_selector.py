"""
src/llm_selector.py  (LUỒNG RIÊNG cho fast_retrieval --llm-select)

LLM đóng vai BỘ CHỌN (selector), KHÔNG sinh đáp án:
  đưa cho LLM câu hỏi + danh sách ứng viên (đã rerank, đánh số), bắt LLM trả về SỐ THỨ TỰ
  của những điều luật trực tiếp làm căn cứ. Output cực ngắn (vài con số) -> nhanh hơn nhiều
  so với sinh answer đầy đủ (2048 token).

Mục tiêu: tăng PRECISION so với "top-2 rerank mù" bằng cách cho LLM lọc trong pool lớn hơn.
LƯU Ý: phải đo lại trên ground_truth_50 — cơ chế này CHƯA chắc thắng top-2 (cần kiểm chứng).
"""
import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

SELECT_SYSTEM = (
    "Bạn là chuyên gia pháp luật Việt Nam. Nhiệm vụ: từ danh sách điều luật ứng viên, "
    "chọn ra những điều TRỰC TIẾP làm căn cứ trả lời câu hỏi. "
    "Chỉ chọn điều thật sự liên quan, ưu tiên độ chính xác hơn số lượng."
)


def select_candidates(
    question: str,
    ranked: List[Dict],
    pipe,
    tokenizer,
    pool_k: int = 6,
    max_select: int = 2,
) -> List[Dict]:
    """
    Trả về danh sách context (con) được LLM chọn, theo thứ tự ưu tiên của LLM.
    Fallback: nếu LLM không trả về số hợp lệ -> lấy top theo rerank.
    """
    pool = ranked[:pool_k]
    if not pool:
        return []

    lines = []
    for i, c in enumerate(pool, 1):
        md = c.get("metadata", {})
        article = md.get("article_id", "")
        doc_number = md.get("doc_number", "")
        title = (md.get("title", "") or "")[:50]
        snippet = re.sub(r"\s+", " ", (c.get("text", "") or "")).strip()[:180]
        lines.append(f"[{i}] {article} — {doc_number} ({title}): {snippet}")

    user_prompt = (
        f"CÂU HỎI: {question}\n\n"
        f"DANH SÁCH ĐIỀU LUẬT ỨNG VIÊN:\n" + "\n".join(lines) +
        f"\n\nHãy chọn TỐI ĐA {max_select} điều TRỰC TIẾP làm căn cứ trả lời câu hỏi, "
        f"quan trọng nhất trước. CHỈ TRẢ LỜI bằng các SỐ THỨ TỰ trong [1-{len(pool)}] "
        f"cách nhau bởi dấu phẩy (ví dụ: 2,5). Tuyệt đối không giải thích, không viết chữ."
    )

    messages = [
        {"role": "system", "content": SELECT_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = f"{SELECT_SYSTEM}\n\n{user_prompt}\n\nTrả lời:"

    try:
        out = pipe(prompt, max_new_tokens=24, do_sample=False)
        gen = out[0]["generated_text"]
        ans = gen[len(prompt):] if prompt in gen else gen.split("Trả lời:")[-1]
    except Exception as e:
        logger.warning(f"[llm_selector] LLM lỗi, fallback top-rerank: {e}")
        return pool[:max_select]

    # Bóc các số thứ tự LLM chọn, giữ thứ tự, loại trùng, trong khoảng hợp lệ
    chosen: List[Dict] = []
    seen = set()
    for n in re.findall(r"\d+", ans):
        idx = int(n)
        if 1 <= idx <= len(pool) and idx not in seen:
            seen.add(idx)
            chosen.append(pool[idx - 1])
        if len(chosen) >= max_select:
            break

    if not chosen:
        # LLM không trả số hợp lệ -> giữ top theo rerank để không mất recall
        chosen = pool[:max_select]

    return chosen
