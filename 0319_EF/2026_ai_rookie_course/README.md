# 2026 AI Rookie Course

歡迎來到 **2026 AI Rookie Course**。本課程旨在引導學員從基礎到進階，掌握當前生成式 AI 領域中最重要的兩大技術：**Embedding 微調 (Finetune)** 與 **檢索增強生成 (RAG)**。

本專案包含兩個核心工作坊，每個工作坊都提供了一系列循序漸進的實驗 (Labs)，幫助您建立完整的技術體系。

---

## 📂 課程目錄

### 1. [Embedding Finetune 實作工作坊](./embedding_finetune/)
專注於 Embedding 技術的核心概念與模型微調。
- **重點內容**：向量相似度、MTEB 評估框架、對比學習 (Contrastive Learning)、Hard Negative Mining、模型微調實作、稀疏表示 (Sparse Representations)。
- **適用場景**：當預訓練模型無法準確捕捉領域知識，或需要提升檢索精度時。

### 2. [RAG 實作工作坊](./RAG/)
專注於構建高效、精準的檢索增強生成系統。
- **重點內容**：文件處理、向量資料庫 (ChromaDB/FAISS)、混合搜索 (Hybrid Search)、進階優化 (Reranking/Query Rewriting)、系統評估 (RAGAS)。
- **適用場景**：構建企業級知識庫問答、對話機器人或需要結合私有資料的 AI 應用。

---

## 🛠️ 環境快速建置

本課程所有實驗均可於**本地環境**執行，建議配備至少 16GB RAM。

### 1. 安裝 Ollama (推薦)
部分實驗 (如 Lab 03 資料生成與 RAG 模型推論) 使用 [Ollama](https://ollama.com/) 運行本地 LLM。請先下載並安裝。

### 2. 設定 Python 環境
建議為每個工作坊建立獨立的虛擬環境。

#### Embedding Finetune
```bash
cd embedding_finetune
python -m venv venv
# Windows: .\venv\Scripts\activate | macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

#### RAG
```bash
cd RAG
python -m venv venv
# Windows: .\venv\Scripts\activate | macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

---

## 📖 學習建議路徑

1. **從 Embedding 開始**：先了解資料如何轉化為向量，以及如何評估向量的品質。
2. **深入微調**：學習如何針對特定任務優化 Embedding 模型。
3. **實作 RAG**：將優化後的向量技術應用到完整的問答系統中。
4. **評估與優化**：學習如何量化系統效能並持續改良。

---

*2026 AI Rookie Course - 讓我們一起探索 AI 的無限可能！*
