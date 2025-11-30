# Vietnamese Food Ordering Chatbot
## LLM + RAG + Reranker System

A complete implementation of a Vietnamese food ordering chatbot using modern NLP techniques: Large Language Models (LLM), Retrieval-Augmented Generation (RAG), and Reranking.

The repository is available [here](https://github.com/Pinminh/food-chatbot).

---

## Features

- **Open-source LLM**: Uses Qwen2.5-3B-Instruct (lightweight, multilingual)
- **RAG System**: Retrieval-Augmented Generation with FAISS vector database
- **Reranker**: BAAI/bge-reranker-v2-m3 for improved relevance
- **Two Modes**: Batch processing and interactive chat
- **Performance Evaluation**: Built-in metrics (F1, BLEU, ROUGE)

---

## Project Structure

```
.
├── main.py                       # Main entry point
├── config.py                     # Configuration settings
├── chatbot.py                    # Chatbot system
├── rag_system.py                 # RAG + reranker
├── llm_generator.py              # LLM response generation
├── data_loader.py                # Data loading utilities
├── evaluator.py                  # Performance evaluation
├── pyproject.toml                # Python dependencies
├── requirements.txt
├── uv.lock
├── Dockerfile                    # Docker configuration
├── README.md                     # This file
├── data/
│   ├── menu.json                 # Vietnamese food menu in json
│   └── menu.txt                  # Vietnamese food menu in txt
├── input/
│   ├── queries.txt               # User queries
│   └── answers.txt               # Ground-truth answers
└── output/
    ├── results.json              # Full results
    ├── answers.txt               # Answers only
    ├── formatted_output.txt      # Query-context-response outputs
    ├── query_response_pairs.txt  # Query-answer pairs
    └── evaluation_metrics.json   # metrics
```

---

## Quick Start

A working Colab notebook to run the system is available [here](https://colab.research.google.com/drive/1rYZk_YzPcdsjYIEoJ1X11Xc0Bb0me3ia?usp=sharing).

### Option 1: Local Installation

#### Prerequisites
- Python 3.8
- 8GB+ RAM (16GB recommended)
- GPU optional (T4 or better for faster inference)

#### Installation

1. **Clone or extract the project**

2. **Install dependencies**
```bash
pip install -r requirements.txt
```
or if you have installed `uv`:
```bash
uv sync
```

3. **Run the chatbot**

**Batch evaluation**:
```bash
python main.py --mode batch --evaluate
```

**Interactive mode**:
```bash
python main.py --mode interactive
```

---

### Option 2: Docker

#### Build Docker image
```bash
docker build -t food-chatbot .
```

#### Run with Docker
```bash
docker run --rm --gpus all -v $(pwd)/output:/app/output food-chatbot
```

For interactive mode:
```bash
docker run --rm -it --gpus all food-chatbot --mode interactive
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        USER QUERY                           │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
         ┌────────────────┐
         │  Data Loader   │
         │  (Menu Data)   │
         └────────┬───────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    RAG SYSTEM                               │
│                                                             │
│  ┌───────────────┐   ┌──────────────┐   ┌──────────────┐    │
│  │   Embedding   │──>│   FAISS      │──>│   Reranker   │    │
│  │   (BGE-M3)    │   │   Retrieval  │   │  (BGE-v2-m3) │    │
│  └───────────────┘   └──────────────┘   └──────┬───────┘    │
│                                                │            │
│                                          Top-K Documents    │
└────────────────────────────────────────────────┼────────────┘
                                                 │
                                                 ▼
                                          ┌─────────────┐
                                          │   Context   │
                                          └──────┬──────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    LLM GENERATOR                            │
│                                                             │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Qwen2.5-3B-Instruct                               │     │
│  │  Query + Context → Natural Response                │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │  FINAL RESPONSE   │
                    └───────────────────┘
```

### LLM: Qwen2.5-3B-Instruct
- **Size**: 3B parameters
- **Language**: Multilingual (including Vietnamese)
- **Advantage**: Lightweight, runs on T4 GPU or CPU
- **Source**: [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)

### Embeddings: BAAI/bge-m3
- **Type**: Dense retrieval embeddings
- **Language**: Multilingual
- **Dimension**: 1024
- **Source**: [Hugging Face](https://huggingface.co/BAAI/bge-m3)

### Reranker: BAAI/bge-reranker-v2-m3
- **Type**: Cross-encoder reranker
- **Language**: Multilingual (including Vietnamese)
- **Advantage**: Improved relevance scoring
- **Source**: [Hugging Face](https://huggingface.co/BAAI/bge-reranker-v2-m3)

---

## Usage Examples

### Batch Mode

Place your queries in `input/queries.txt` (one per line):
```
Có những món nào trong menu?
Phở bò giá bao nhiêu?
Tôi muốn đặt 2 phần phở bò và 1 ly trà sữa
```

Run:
```bash
python main.py --mode batch --evaluate
```

### Interactive Mode

```bash
python main.py --mode interactive
```

Example conversation:
```
Bạn: Có món gà rán không?
Chatbot: Có ạ, chúng tôi có món Gà rán với giá 60,000đ...

Bạn: Giá bao nhiêu?
Chatbot: Món gà rán có giá 60,000đ...
```

---

## Performance Evaluation

The system evaluates performance using multiple metrics:

### Metrics
- **F1 Score**: Token-level overlap
- **BLEU**: N-gram precision
- **ROUGE-L**: Longest common subsequence

### Context Retrieval
- Retrieval success rate
- Average response length

Results are saved to `output/evaluation_metrics.json`

---

## Configuration

Edit `config.py` to customize:

```python
# Model selection
LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# RAG parameters
TOP_K_RETRIEVAL = 10  # Initial retrieval
TOP_K_RERANK = 3      # After reranking

# LLM generation
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9
DO_SAMPLE = False
```

---

## Output Files

### results.json
Complete results with query, context, and response:
```json
[
  {
    "query": "Phở bò giá bao nhiêu?",
    "context": "...",
    "response": "Phở bò có giá 50,000đ..."
  }
]
```

### answers.txt
One answer per line (for evaluation):
```
Phở bò có giá 50,000đ...
Có ạ, chúng tôi có món gà rán...
```

### formatted_output.txt
Human-readable format with context.

### query_response_pairs.txt
Q&A pairs format.

### evaluation_metrics.json
Performance metrics and statistics.

---

## Assignment Information

**Course**: CO3085 - Natural Language Processing  
**Semester**: 1, Academic Year 2025-2026  
**Part**: 2 - Chatbot with LLM/RAG  
**Option**: 2 - Full chatbot with LLM + RAG + Reranker

---

**Note**: First run will download models (~3-5GB) and install heavy dependencies. Ensure stable internet connection.
