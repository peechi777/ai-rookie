from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot

MODEL_PATH = "/home/user/model/Llama-3.2-1B-Instruct" 

recipe = [
    GPTQModifier(
        scheme="W8A8",        # 改回 W8A8，這對 4070 比較友善
        targets="Linear", 
        ignore=["lm_head"]    # 忽略輸出層即可
    ),
]

oneshot(
    model=MODEL_PATH,
    dataset="open_platypus",
    recipe=recipe,
    output_dir="./Llama-3.2-1B-Instruct-W8A8-HighPrec", 
    # --- 提升精度的關鍵 ---
    max_seq_length=2048,           # 從 512 改回 2048（讓模型考慮更長的上下文）
    num_calibration_samples=1024   # 從 256 改回 1024（用更多數據來校準權重）
)