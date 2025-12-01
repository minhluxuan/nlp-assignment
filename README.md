# Vietnamese Food Ordering Chatbot
## Hệ thống LLM + RAG + Reranker

Một giải pháp hoàn chỉnh cho chatbot đặt món ăn Việt Nam, được xây dựng dựa trên các kỹ thuật NLP hiện đại: Mô hình ngôn ngữ lớn (LLM), Retrieval-Augmented Generation (RAG), và Reranking.

Repo dự án tại đây [here](https://github.com/Pinminh/food-chatbot).

---

## Đặc trưng dự án:

- **LLM mã nguồn mở**: Sử dụng Qwen2.5-3B-Instruct (nhẹ, đa ngôn ngữ)
- **Hệ thống RAG**: Tìm kiếm kết hợp sinh câu trả lời bằng FAISS vector database
- **Reranker**: Áp dụng BAAI/bge-reranker-v2-m3 để tăng độ phù hợp
- **Hai chế độ chạy**: Xử lý hàng loạt và trò chuyện tương tác
- **Đánh giá chất lượng**: Tích hợp các chỉ số F1, BLEU, ROUGE

---

## Cấu trúc dự án

```
.
├── main.py                      # Entry point của hệ thống
├── config.py                    # Cấu hình chung
├── chatbot.py                   # Logic chatbot
├── rag_system.py                # RAG + reranker
├── llm_generator.py             # Sinh phản hồi từ LLM
├── data_loader.py               # Tiện ích load dữ liệu
├── evaluator.py                 # Đánh giá hiệu suất
├── pyproject.toml               # Cấu hình project
├── requirements.txt
├── uv.lock
├── Dockerfile                   # Cấu hình Docker
├── README.md
│
├── data/
│   ├── menu.json                # Menu món ăn (JSON)
│   └── menu.txt                 # Menu món ăn (text)
│
├── input/
│   ├── queries.txt              # Các câu hỏi đầu vào
│   └── answers.txt              # Đáp án chuẩn
│
└── output/
    ├── results.json             # Kết quả chi tiết
    ├── answers.txt              # Chỉ câu trả lời
    ├── formatted_output.txt     # Query – context – response
    ├── query_response_pairs.txt # Cặp query – response
    └── evaluation_metrics.json  # Các chỉ số đánh giá

```

---

## Quick Start

Bạn có thể chạy hệ thống trực tiếp bằng Colab thông qua notebook tại liên kết sau:
https://colab.research.google.com/drive/1rYZk_YzPcdsjYIEoJ1X11Xc0Bb0me3ia?usp=sharing.

Video minh họa sử dụng hệ thống có tại: 

### Option 1: Chạy ở máy local

#### Yêu cầu tiên quyết
- Python 3.8
- 8GB+ RAM (ít nhất 16GB)
- GPU (có hoặc không) (T4 là vừa đủ)

#### Cài đặt

1. **Clone hoặc giải nén dự án về máy**

2. **Cài đặt thư viện và phụ thuộc**
```bash
pip install -r requirements.txt
```
hoặc nếu có `uv`:
```bash
uv sync
```

3. **Chạy chatbot**
```bash
python main.py --mode interactive
```

---

### Option 2: Docker

#### Build Docker image
```bash
docker build -t food-chatbot .
```

#### Chạy bằng Docker
```bash
docker run --rm -it --gpus all food-chatbot --mode interactive
```

---

### Mô hình ngôn ngữ lớn được sử dụng: Qwen2.5-3B-Instruct
- **Số lượng tham số**: 3 tỉ
- **Ngôn ngữ**: Đa ngôn ngữ (bao gồm tiếng Việt)
- **Điểm mạnh**: Nhẹ, chạy được trên GPU T4 hoặc CPU
- **Nguồn**: [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)

### Embeddings: BAAI/bge-m3
- **Loại**: Dense retrieval embeddings
- **Ngôn ngữ**: Multilingual
- **Kích thước**: 1024
- **Nguồn**: [Hugging Face](https://huggingface.co/BAAI/bge-m3)

### Reranker: BAAI/bge-reranker-v2-m3
- **Loại**: Cross-encoder reranker
- **Ngôn ngữ**: Multilingual (including Vietnamese)
- **Điểm mạnh**: Improved relevance scoring
- **Nguồn**: [Hugging Face](https://huggingface.co/BAAI/bge-reranker-v2-m3)

---

## Usage Examples
```bash
python main.py --mode interactive
```

Ví dụ về cuộc hội thoại:
```
Bạn: Có món bún đậu không?
Chatbot: Có ạ, chúng tôi có món Bún đậu với giá 110,000đ...

Bạn: Giá bao nhiêu?
Chatbot: Món bún đậu có giá 60,000đ...
```

---

## Đánh Giá Hiệu Suất

Hệ thống đánh giá hiệu suất thông qua nhiều chỉ số:

### Các Chỉ Số
- **F1 Score**: Độ trùng khớp ở mức token
- **BLEU**: Độ chính xác n-gram
- **ROUGE-L**: Chuỗi con chung dài nhất

### Truy Xuất Ngữ Cảnh
- Tỷ lệ truy xuất thành công
- Độ dài phản hồi trung bình

Kết quả được lưu vào `output/evaluation_metrics.json`

---

## Cấu Hình

Chỉnh sửa `config.py` để tùy chỉnh:
```python
# Lựa chọn mô hình
LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Tham số RAG
TOP_K_RETRIEVAL = 10  # Truy xuất ban đầu
TOP_K_RERANK = 3      # Sau khi xếp hạng lại

# Sinh văn bản LLM
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9
DO_SAMPLE = False
```

---

## Các File Đầu Ra

### results.json
Kết quả đầy đủ bao gồm câu hỏi, ngữ cảnh và câu trả lời:
```json
[
  {
    "query": "Chè ba màu giá bao nhiêu?",
    "context": "...",
    "response": "Chè ba màu có giá 30,000đ..."
  }
]
```

### answers.txt
Mỗi câu trả lời trên một dòng (để đánh giá):
```
Chè ba màu có giá 50,000đ...
Có ạ, chúng tôi có món bún đậu mắm tôm tá lả...
```

### formatted_output.txt
Định dạng dễ đọc với ngữ cảnh.

### query_response_pairs.txt
Định dạng cặp câu hỏi-đáp.

### evaluation_metrics.json
Các chỉ số hiệu suất và thống kê.

---

## Thông Tin Bài Tập

**Môn học**: CO3085 - Xử Lý Ngôn Ngữ Tự Nhiên  
**Học kỳ**: 1, Năm học 2025-2026  
**Phần**: 2 - Chatbot với LLM/RAG  
**Lựa chọn**: 2 - Chatbot hoàn chỉnh với LLM + RAG + Reranker

---

**Lưu ý**: Lần chạy đầu tiên sẽ tải xuống các mô hình (~3-5GB) và cài đặt các thư viện nặng. Đảm bảo kết nối internet ổn định.