import json

def get_kv_cache_info(config, seq_len):

    layers = config.get("num_hidden_layers") #模型總層數
    kv_heads = config.get("num_key_value_heads") 
    head_dim = config.get("head_dim") or (config.get("hidden_size") // config.get("num_attention_heads"))

    is_swa = config.get("use_sliding_window", False) and config.get("sliding_window")
    k_len = min(seq_len, config.get("sliding_window")) if is_swa else seq_len
    
    gb_size = (2 * layers * kv_heads * head_dim * 2 * k_len) / (1024**3)
    
    return is_swa, k_len, gb_size

with open("config.json", "r") as f:
    config_data = json.load(f)

is_swa, k, size = get_kv_cache_info(config_data, 32768)

print(f"Is SWA: {is_swa}")
print(f"Token length K: {k}")
print(f"KV Cache Size: {size:.4f} GB")