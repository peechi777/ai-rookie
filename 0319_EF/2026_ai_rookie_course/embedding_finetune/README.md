# 2026 AI Rookie Course - Embedding Finetune 實作工作坊

這個資料夾包含 **Embedding Finetune** 技術的系列實驗。透過本系列實作，您將學習如何理解、評估、以及微調 embedding 模型，並探索進階的稀疏表示技術。

## 🚀 專案簡介

本課程旨在引導學員深入掌握 embedding 技術的核心概念與實務應用。從基礎的向量表示到進階的 finetune 技術，循序漸進地建立完整的知識體系。

## 📂 目錄結構

```
embedding_finetune/
├── embedding_finetune_lab01.ipynb  # Embedding 基礎與視覺化
├── embedding_finetune_lab02.ipynb  # 評估指標與基準測試
├── embedding_finetune_lab03.ipynb  # Contrastive Learning 與資料準備
├── embedding_finetune_lab04.ipynb  # Finetune 實作
├── embedding_finetune_lab05.ipynb  # Sparse Representations (Seismic)
├── requirements.txt                 # 套件依賴
└── README.md                        # 本說明文件
```

## 📖 實驗說明

### Lab 01: Embedding 基礎與視覺化
- 理解什麼是 Embedding 以及為何需要它
- 掌握向量相似度計算 (Cosine, Euclidean, Dot Product)
- 使用 t-SNE / UMAP 視覺化 embedding space
- 比較不同預訓練模型的 embedding 品質

### Lab 02: 評估指標與基準測試
- 學習 Retrieval 評估指標：MRR, NDCG, Recall@K, Precision@K
- 認識 MTEB (Massive Text Embedding Benchmark) 框架
- 建立自己的評估 pipeline
- 量化比較不同模型的效能

### Lab 03: Contrastive Learning 與訓練資料準備
- 理解 Contrastive Learning 原理 (InfoNCE Loss)
- 掌握 Hard Negative Mining 的重要性
- 學習從原始資料生成訓練三元組
- 使用 LLM 自動生成合成訓練資料

### Lab 04: Embedding Finetune 實作
- 使用 `sentence-transformers` 進行完整 finetune
- 比較不同 Loss Functions (MultipleNegativesRankingLoss, TripletLoss)
- 實作 Early Stopping 與 Learning Rate Scheduling
- 評估 finetune 前後的效能差異

### Lab 05: Learned Sparse Representations 與 Seismic
- 理解 Dense vs Sparse Embedding 的差異與優劣
- 學習 SPLADE 等 Learned Sparse 模型原理
- 使用 Seismic 建立高效稀疏向量索引
- 實作 Hybrid 檢索 (結合 Dense 和 Sparse)

## 🛠️ 環境建置

### 1. 前置需求
- Python 3.10+
- GPU 推薦 (有 8GB+ VRAM 為佳，CPU 也可以執行但較慢)
- [Ollama](https://ollama.com/) (選用，用於 Lab 03 的合成資料生成)

### 2. 安裝套件

建議使用虛擬環境執行：

```bash
# 建立虛擬環境 (選用)
python -m venv venv

# 啟動虛擬環境 (Windows)
.\venv\Scripts\activate

# 啟動虛擬環境 (macOS/Linux)
source venv/bin/activate

# 安裝所需套件
pip install -r requirements.txt
```

### 3. 驗證安裝

```python
import torch
from sentence_transformers import SentenceTransformer

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Sentence Transformers 載入成功！")
```

## 📊 學習路徑

```
┌─────────────────────────────────────────────────────────────┐
│                    學習路徑圖                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Lab 01: 基礎          Lab 02: 評估                         │
│  ┌──────────┐          ┌──────────┐                         │
│  │ Embedding │ ──────► │ MRR/NDCG │                         │
│  │ 相似度    │          │ MTEB     │                         │
│  └──────────┘          └────┬─────┘                         │
│                             │                               │
│                             ▼                               │
│  Lab 03: 資料準備      Lab 04: Finetune                     │
│  ┌──────────┐          ┌──────────┐                         │
│  │ Hard Neg │ ──────► │ Training │                         │
│  │ 合成資料  │          │ Loss Fn  │                         │
│  └──────────┘          └────┬─────┘                         │
│                             │                               │
│                             ▼                               │
│                    Lab 05: 進階                             │
│                    ┌──────────┐                             │
│                    │ Sparse   │                             │
│                    │ Hybrid   │                             │
│                    └──────────┘                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 技術棧

| 類別 | 工具 |
|------|------|
| Embedding Models | `sentence-transformers`, `transformers` |
| Sparse Models | SPLADE via `transformers` |
| Sparse Index | `pyseismic-lsr` (Seismic) |
| Evaluation | `mteb`, 自建 evaluator |
| Visualization | `matplotlib`, `umap-learn`, `seaborn` |
| LLM (合成資料) | Ollama (選用) |

## ⚠️ 注意事項

- 本實驗建議電腦具備至少 16GB RAM
- 若無 GPU，執行部分 Embedding 或訓練時可能會較慢，但均可在 CPU 上運行
- Lab 03 的 LLM 合成資料功能需要 Ollama，如未安裝會使用備用方案
- Lab 05 的 Seismic 需要 `pyseismic-lsr` 套件，請確保已正確安裝

## 📚 參考資源

### 論文
- [Sentence-BERT](https://arxiv.org/abs/1908.10084) - Sentence Embeddings using Siamese BERT-Networks
- [SPLADE](https://arxiv.org/abs/2107.05720) - Sparse Lexical and Expansion Model
- [Seismic (SIGIR 2024)](https://doi.org/10.1145/3626772.3657769) - Efficient Inverted Indexes for Learned Sparse Representations

### 工具文件
- [Sentence Transformers](https://www.sbert.net/)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [Seismic GitHub](https://github.com/TusKANNy/seismic)

### 教學資源
- [Hugging Face Embedding Course](https://huggingface.co/learn)
- [Pinecone Learning Center](https://www.pinecone.io/learn/)

---

*2026 AI Rookie Course - 讓我們一起探索 Embedding 的無限可能！*
