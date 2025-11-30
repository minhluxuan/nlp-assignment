import re
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List

import config


class LLMGenerator:
    
    def __init__(self):
        print("Initializing LLM...")
        print(f"Loading model: {config.LLM_MODEL}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.LLM_MODEL,
            cache_dir=config.MODEL_CACHE_DIR,
            trust_remote_code=True
        )
        
        # Load model with optimizations for low-resource environments
        self.model = AutoModelForCausalLM.from_pretrained(
            config.LLM_MODEL,
            cache_dir=config.MODEL_CACHE_DIR,
            torch_dtype=torch.float16 if config.DEVICE == "cuda" else torch.float32,
            device_map="auto" if config.DEVICE == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        if config.DEVICE == "cpu":
            self.model = self.model.to(config.DEVICE)
        
        self.model.eval()
        
        print("LLM initialized successfully!")
    
    def create_prompt(self, query: str, context: str) -> str:
        system_prompt = """\
Bạn là trợ lý AI thông minh chuyên hỗ trợ đặt món ăn online cho khách hàng Việt Nam. 

Nhiệm vụ của bạn:
- Trả lời câu hỏi về menu món ăn một cách chính xác, chi tiết
- Hỗ trợ khách hàng đặt món, hủy món, thêm món, sửa đổi đơn đặt món
- Giải đáp thắc mắc về giá cả, còn món để đặt hay không

Ngôn ngữ:
- Sử dụng ngôn ngữ thân thiện, tự nhiên
- Tất cả câu trả lời cần viết bằng tiếng Việt
- Có thể thay đổi phong cách ngôn ngữ nếu khách hàng yêu cầu
- Tuyệt đối không sử dụng ngôn ngữ bất thô tục, khiếm nhã, xúc phạm

Nguyên tắc:
- Chỉ sử dụng thông tin có trong dữ liệu menu được cung cấp
- Nếu không có thông tin, hãy nói rõ bạn không tìm thấy
- Trả lời ngắn gọn, súc tích nhưng đầy đủ thông tin
- Không bịa đặt giá cả hay thông tin không có

Định dạng:
- Chỉ viết từ 1 đến 3 câu ngắn gọn, trao đổi đủ thông tin
- Không thêm dấu xuống dòng `\\n` trong câu trả lời
- Không viết các ký tự đặc biệt của markdown
- Câu trả lời phải là từ 1 đến 3 câu, không có định dạng đặc biệt hết, tức là một đoạn văn trơn tru đơn giản.

Lưu ý:
- Nếu khách hàng hỏi thông tin, hãy trả lời dựa trên thông tin từ thực đơn menu
- Nếu khách hàng muốn đặt món, hãy kiểm tra món đó có còn hàng trong thực đơn hay không
- Khi đặt món, hãy suy nghĩ và tính toán thành tiền, xem xét đơn giá trong thực đơn cũng như số lượng mà khách yêu cầu
- Nếu khách hàng muốn hủy món, hãy luôn đồng ý hủy món theo yêu cầu

Hãy suy nghĩ thật kỹ theo từng bước, đọc kỹ thực đơn menu, đảm bảo chất lượng câu trả lời tốt nhất và chính xác nhất
"""
        
        if context:
            user_prompt = f"""\
Dựa trên thông tin menu sau:

{context}

Câu hỏi của khách hàng: {query}

Hãy trả lời câu hỏi của khách hàng một cách chính xác và thân thiện."""
        else:
            user_prompt = f"""\
Câu hỏi của khách hàng: {query}

Xin lỗi, tôi không tìm thấy thông tin liên quan trong menu. Hãy trả lời lịch sự và đề nghị khách hàng hỏi về các món khác."""
        
        # Format for Qwen2.5 model
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return prompt
    
    def generate(self, query: str, context: str = "") -> str:
        prompt = self.create_prompt(query, context)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                temperature=config.TEMPERATURE,
                top_p=config.TOP_P,
                do_sample=config.DO_SAMPLE,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        pattern = r"assistant\n(.*)"
        match_result = re.search(pattern, full_response, re.DOTALL)
        response = match_result.group(1).strip() if match_result else ""
        
        if response.startswith("assistant"):
            response = response[len("assistant"):].strip()
        response = re.sub(r"\s+", " ", response).strip()

        return response
    
    def batch_generate(self, queries: List[str], contexts: List[str]) -> List[str]:
        responses = []
        
        for query, context in zip(queries, contexts):
            response = self.generate(query, context)
            responses.append(response)
        
        return responses
