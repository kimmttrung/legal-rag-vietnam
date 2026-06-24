import re
import logging
import time
from typing import List, Dict, Optional, Tuple

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)

from config.settings import Settings

logger = logging.getLogger(__name__)

# =========================================================
# SYSTEM PROMPT — Ép mô hình trả lời văn xuôi, gọn gàng, không chia mục
# =========================================================
SYSTEM_PROMPT = """Bạn là một robot trợ lý ảo tư vấn pháp luật chuyên nghiệp cho khối doanh nghiệp vừa và nhỏ (SME) tại Việt Nam.

NHIỆM VỤ: Hãy trả lời câu hỏi pháp lý của người dùng một cách chính xác, ngắn gọn và khách quan.
ĐỊNH DẠNG ĐẦU RA BẮT BUỘC:
- Trả lời trực tiếp bằng một đoạn văn duy nhất (tuyệt đối không xuống dòng, không dùng ký tự \\n).
- Không chia tiêu đề (không có "## 1. KẾT LUẬN", "## 2. CĂN CỨ PHÁP LÝ", v.v.).
- Không tự bịa đặt thông tin nằm ngoài phần [NGỮ CẢNH PHÁP LÝ] được cung cấp.
- Chỉ trả lời bằng tiếng Việt, không dùng tiếng Anh."""

def build_user_prompt(query: str, contexts: List[Dict]) -> str:
    context_parts = []
    total_chars = 0

    for i, doc in enumerate(contexts, start=1):
        text = doc.get("text", "")
        metadata = doc.get("metadata", {})

        doc_number = metadata.get("doc_number", "")
        doc_name = metadata.get("doc_name", "")
        article = metadata.get("article", "")

        header = f"[Văn bản {i}]"
        if doc_name:
            header += f" {doc_name}"
        if doc_number:
            header += f" - Số hiệu: {doc_number}"
        if article:
            header += f" - {article}"

        chunk = f"{header}\n{text}"

        if total_chars + len(chunk) > Settings.MAX_CONTEXT_CHARS:
            remaining = Settings.MAX_CONTEXT_CHARS - total_chars
            if remaining > 200:
                chunk = chunk[:remaining] + "...[cắt bớt]"
                context_parts.append(chunk)
            break

        context_parts.append(chunk)
        total_chars += len(chunk)

    context_text = "\n\n" + "\n\n---\n\n".join(context_parts) + "\n"

    prompt = f"""[NGỮ CẢNH PHÁP LÝ]
{context_text}
[CÂU HỎI]
{query}

[YÊU CẦU]
Dựa hoàn toàn vào [NGỮ CẢNH PHÁP LÝ], hãy trả lời câu hỏi trên bằng một đoạn văn ngắn gọn, viết liền mạch không xuống dòng."""

    return prompt


class AnswerGenerator:
    def __init__(self):
        logger.info(f"[LLM] Đang load model: {Settings.LLM_MODEL_NAME}")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            Settings.LLM_MODEL_NAME,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            Settings.LLM_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        self.model.eval()

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto"
        )
        logger.info("✅ LLM load thành công!")

    def _extract_references(self, contexts: List[Dict]) -> Tuple[List[str], List[str]]:
        """
        Trích xuất tự động và chuẩn hóa trường relevant_docs và relevant_articles từ metadata của context.
        Định dạng:
        - relevant_docs: ["mã văn bản|tên văn bản"]
        - relevant_articles: ["mã văn bản|tên văn bản|điều"]
        """
        relevant_docs_set = set()
        relevant_articles_set = set()

        for doc in contexts:
            metadata = doc.get("metadata", {})
            doc_number = metadata.get("doc_number", "").strip()
            doc_name = metadata.get("doc_name", "").strip()
            article = metadata.get("article", "").strip() # Ví dụ: "Điều 4" hoặc "Điều 5"

            if not doc_number:
                continue

            # Chuẩn hóa tên văn bản (Ví dụ: "Luật 04/2017/QH14 Luật Hỗ trợ doanh nghiệp nhỏ và vừa")
            full_doc_title = doc_name
            if doc_number not in full_doc_title:
                # Nếu trong tên chưa có mã, có thể tự format lại cho giống mẫu bài thi của bạn
                if "luật" in doc_name.lower() and not doc_name.lower().startswith("luật"):
                    full_doc_title = f"Luật {doc_number} {doc_name}"
                elif "nghị định" in doc_name.lower() and not doc_name.lower().startswith("nghị định"):
                    full_doc_title = f"Nghị định {doc_number} {doc_name}"

            # 1. Tạo chuỗi cho relevant_docs
            doc_str = f"{doc_number}|{full_doc_title}"
            relevant_docs_set.add(doc_str)

            # 2. Tạo chuỗi cho relevant_articles (Nếu có thông tin Điều)
            if article:
                # Trích xuất chính xác chữ "Điều X" từ chuỗi metadata
                article_match = re.search(r'(Điều\s+\d+|Điều\s+[A-Z0-9]+)', article, re.I)
                article_clean = article_match.group(0) if article_match else article
                
                article_str = f"{doc_number}|{full_doc_title}|{article_clean}"
                relevant_articles_set.add(article_str)

        return list(relevant_docs_set), list(relevant_articles_set)

    def generate(
        self,
        query: str,
        contexts: List[Dict],
        temperature: float = Settings.LLM_TEMPERATURE
    ) -> Dict:
        """
        Sinh câu trả lời và đóng gói thành một Object hoàn chỉnh theo đúng format cấu trúc bài thi.
        """
        user_prompt = build_user_prompt(query, contexts)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        try:
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            formatted_prompt = f"{SYSTEM_PROMPT}\n\nNgười dùng: {user_prompt}\n\nTrợ lý:"

        # Tăng token lên 2048 để đảm bảo Qwen/Deepseek không bị cụt câu chữ
        max_tokens = max(2048, getattr(Settings, "LLM_MAX_NEW_TOKENS", 2048))
        
        output = self.pipe(
            formatted_prompt,
            max_new_tokens=max_tokens, 
            temperature=temperature,
            top_p=Settings.LLM_TOP_P,
            repetition_penalty=Settings.LLM_REPETITION_PENALTY,
            do_sample=True if temperature > 0 else False,
        )

        generated_text = output[0]["generated_text"]

        # Bóc tách text phản hồi của LLM
        if formatted_prompt in generated_text:
            answer = generated_text[len(formatted_prompt):].strip()
        else:
            answer = generated_text.split("Trợ lý:")[-1].strip()

        # Xóa tàn dư thẻ suy nghĩ nếu cậu vẫn dùng dòng DeepSeek-R1
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
        answer = re.sub(r'<think>.*', '', answer, flags=re.DOTALL).strip()

        # DỌN DẸP TEXT: Thay thế toàn bộ dấu xuống dòng \n thành khoảng trắng để biến thành 1 đoạn văn duy nhất
        answer = re.sub(r'\s+', ' ', answer).strip()

        # Tự động trích xuất cấu trúc văn bản từ nguồn context thực tế của Reranker
        relevant_docs, relevant_articles = self._extract_references(contexts)

        # Trả về đúng cấu trúc của một phần tử trong mảng kết quả bài thi
        return {
            "answer": answer,
            "relevant_docs": relevant_docs,
            "relevant_articles": relevant_articles
        }