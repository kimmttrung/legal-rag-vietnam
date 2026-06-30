"""
src/answer_intersect.py  (LUỒNG RIÊNG cho fast_retrieval --llm-answer)

Pipeline: LLM sinh answer THẬT -> lấy GIAO giữa các điều/số hiệu LLM trích dẫn trong answer
với TOP pool ứng viên rerank. Mục tiêu:
  - relevant_docs/articles bám đúng điều LLM thực sự dùng (grounded trong pool rerank).
  - DERIVE docs TỪ các điều được chọn => mọi văn bản đều có ít nhất 1 Điều (hết "doc mồ côi").
  - answer khác rỗng => còn ăn điểm QA (5 tiêu chí).

LƯU Ý: giao-citation từng cho F2 thấp hơn top-2 rerank trên GT-50 -> BẮT BUỘC đo lại
bằng score_retrieval.py trước khi nộp.
"""
import re
import logging
from typing import List, Dict, Set, Tuple

from config.settings import Settings

logger = logging.getLogger(__name__)

_ARTICLE_NUM_RE = re.compile(r"[Đđ]iều\s+(\d+)")
# Số hiệu VBPL: phần đuôi gồm CHỮ + SỐ + gạch nối, vd "QH13", "NĐ-CP", "TT-BTC", "QĐ-TTg".
# (Bản cũ thiếu \d ở char-class nên "91/2015/QH13" bị cắt còn "91/2015/QH" → so khớp doc luôn trượt.)
_DOC_NUM_RE = re.compile(r"\d{1,4}/\d{4}/[A-Za-zĐđ0-9\-]+")


def parse_citations(answer_text: str) -> Tuple[Set[str], Set[str]]:
    """Bóc các số Điều và số hiệu văn bản LLM nhắc tới trong answer."""
    art_nums = set(_ARTICLE_NUM_RE.findall(answer_text or ""))
    doc_nums = set(_DOC_NUM_RE.findall(answer_text or ""))
    return art_nums, doc_nums


def _article_num(article_id: str) -> str:
    m = re.search(r"(\d+)", article_id or "")
    return m.group(1) if m else ""


def _ctx_key(c: Dict) -> Tuple[str, str]:
    """Khóa định danh 1 context (để khử trùng giữa intersect & union): (số hiệu, số Điều)."""
    md = c.get("metadata", {})
    return (md.get("doc_number", "").strip(), _article_num(md.get("article_id", "")))


def _pure_intersect(answer_text: str, pool: List[Dict]) -> List[Dict]:
    """GIAO thuần: các context trong pool mà LLM có trích Điều (KHÔNG fallback). Có thể rỗng."""
    cited_arts, cited_docs = parse_citations(answer_text)

    def is_cited(c: Dict) -> bool:
        md = c.get("metadata", {})
        anum = _article_num(md.get("article_id", ""))
        if not anum or anum not in cited_arts:
            return False
        # Nếu answer có nêu số hiệu, bắt khớp đúng văn bản để tránh nhầm 'Điều X' giữa các luật
        if cited_docs:
            return md.get("doc_number", "").strip() in cited_docs
        return True

    return [c for c in pool if is_cited(c)]


def intersect_select(
    answer_text: str,
    ranked: List[Dict],
    pool_k: int = 8,
    max_out: int = 5,
) -> List[Dict]:
    """
    Trả về danh sách context (con) = GIAO giữa điều LLM trích dẫn và pool rerank[:pool_k],
    giữ thứ tự rerank, tối đa max_out phần tử.
    Fallback: nếu giao rỗng -> top-N rerank (Settings.RELEVANT_ARTICLES_MAX) để không mất recall.
    """
    pool = ranked[:pool_k]
    if not pool:
        return []

    kept = _pure_intersect(answer_text, pool)

    if not kept:
        # LLM không trích được điều nào khớp pool -> giữ top theo rerank
        kept = pool[: max(1, Settings.RELEVANT_ARTICLES_MAX)]

    return kept[:max_out]


def intersect_union_select(
    answer_text: str,
    ranked: List[Dict],
    pool_k: int = 8,
    min_keep: int = 2,
    max_out: int = 5,
) -> List[Dict]:
    """
    Luồng KẾT HỢP — ƯU TIÊN điều LLM trích, top rerank chỉ BÙ khi thiếu:
        1) Lấy [GIAO: điều LLM trích ∩ pool rerank]  (ưu tiên hàng đầu, giữ thứ tự rerank).
        2) NẾU số điều ở (1) < `min_keep` -> bù thêm top rerank (chưa có) cho ĐỦ `min_keep`.
        3) Cắt còn tối đa `max_out`.

    Ý nghĩa:
    - LLM trích ĐỦ (>= min_keep điều grounded) -> CHỈ lấy điều LLM trích, rerank KHÔNG chen vào
      (precision cao, không phình).
    - LLM trích THIẾU (hoặc không khớp pool) -> rerank bù cho đủ min_keep (giữ recall, không rỗng).
    - `min_keep` cũng đóng vai trò chống output rỗng: đặt >= 1 để luôn có ít nhất 1 điều.
    """
    pool = ranked[:pool_k]
    if not pool:
        return []

    cited = _pure_intersect(answer_text, pool)          # điều LLM trích (có thể rỗng)

    out: List[Dict] = []
    seen: Set[Tuple[str, str]] = set()

    # (1) Ưu tiên điều LLM trích trước
    for c in cited:
        k = _ctx_key(c)
        if k[0] and k not in seen:
            seen.add(k)
            out.append(c)

    # (2) Bù bằng top rerank CHỈ khi chưa đủ min_keep
    if len(out) < min_keep:
        for c in ranked:                                # duyệt theo thứ tự rerank (tốt -> kém)
            k = _ctx_key(c)
            if k[0] and k not in seen:
                seen.add(k)
                out.append(c)
                if len(out) >= min_keep:
                    break

    return out[:max_out]
