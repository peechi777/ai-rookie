# 2026 AI Rookie Course - RAG 實作工作坊

這個儲存庫包含 **2026 AI Rookie Course** 中關於 **RAG (Retrieval-Augmented Generation)** 技術的系列實驗與講義。透過本系列實作，您將學習如何從零開始建構、優化以及評估一個完整的 RAG 系統。

## 🚀 專案簡介

本課程旨在引導學員掌握大語言模型 (LLM) 的檢索增強生成技術。所有實驗皆設計為可於**本地環境**執行，強調隱私與對底層技術的掌控。

## 📂 目錄結構

- `RAG 應用.pptx`: RAG 技術原理與應用場景簡報。
- `RAG_lab01.ipynb`: **RAG 基礎入門** - 學習載入文件、文本分割、Embedding 與向量存儲。
- `RAG_lab02.ipynb`: **向量資料庫深入探討** - 比較不同的向量存儲引擎 (ChromaDB, FAISS, Qdrant)。
- `RAG_lab03.ipynb`: **混合搜索策略** - 結合關鍵字搜索 (BM25) 與語義搜索的混合檢索。
- `RAG_lab04.ipynb`: **進階 RAG 優化** - 學習查詢改寫 (Query Rewriting) 與重新排序 (Reranking)。
- `RAG_lab05.ipynb`: **系統評估** - 使用 RAGAS 框架對 RAG 系統進行量化評估。
- `requirements.txt`: 專案所需的 Python 套件清單。
- `test_run.py`: 用於環境測試與資料準備的腳本。

## 🛠️ 環境建置

### 1. 前置需求
- Python 3.10+
- [Ollama](https://ollama.com/): 用於在本地執行大型語言模型（建議安裝後先下載 `llama3.2` 或 `gemma2` 模型）。

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

### 3. 資料準備
在開始 Lab 01 之前，可以執行測試腳本來確保環境正確並下載範例資料：

```bash
python test_run.py
```

## 📖 實驗說明

1. **Lab 01: Hello World RAG**
   - 建立第一個 RAG Pipeline。
   - 使用 `sentence-transformers` 進行本地 Embedding。
   - 整合 `ChromaDB` 與 `Ollama` 進行問答。

2. **Lab 02: Vector DB Deep Dive**
   - 深入理解向量檢索原理。
   - 實作向量的持久化儲存與檢索優化。

3. **Lab 03: Hybrid Search**
   - 解決語義搜索在特定術語檢索上的不足。
   - 實作稀疏向量 (BM25) 與稠密向量 (Embedding) 的權重混合。

4. **Lab 04: Advanced RAG**
   - 引入 Multi-query 與 Reranker 模型。
   - 顯著提升檢索的準確度與相關性。

5. **Lab 05: RAG Evaluation**
   - 學習 RAG 系統的四個關鍵指標：Faithfulness, Answer Relevance, Context Precision, Context Recall。
   - 使用 RAGAS 進行自動化評估。

## ⚠️ 注意事項
- 本實驗建議電腦具備至少 16GB RAM。
- 若無 GPU，執行部分 Embedding 或 LLM 推論時可能會較慢，但均可在 CPU 上運行。
- 請確保 Ollama 服務已在背景啟動。

---
*2026 AI Rookie Course - 讓我們一起探索 AI 的無限可能！*
