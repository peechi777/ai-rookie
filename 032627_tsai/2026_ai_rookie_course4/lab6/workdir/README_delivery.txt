========================================
SFT 指令微調模型交付說明 (Customer Service AI)
========================================

1. 基礎模型資訊
   - Base Model: Qwen/Qwen2.5-3B-Instruct
   - 微調技術: QLoRA (4-bit Quantization)

2. 檔案目錄
   - adapter/ : 包含 LoRA 權重與 tokenizer 配置文件
   - inference.py : 獨立推理腳本
   - test.jsonl : 原始測試資料集

3. 執行環境
   - 安裝依賴: uv sync
   - 硬體需求: NVIDIA GPU (建議 6GB VRAM 以上)

4. 推理指令範例
   python inference.py --user "我想修改我的收件地址，請問該怎麼操作？" --adapter_dir "adapter"

5. 注意事項
   - 務必使用 tokenizer.apply_chat_template 進行推理，否則回答品質會大幅下降。
   - 本模型已針對禮貌用語與結構化回答進行優化。
