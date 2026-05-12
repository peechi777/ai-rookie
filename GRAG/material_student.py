import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import numpy as np

def Softmax( logits:torch.Tensor, temperature:float=1.0 ):
    print( f"[INFO][SOFTMAX] using SOFTMAX with T={temperature}" )

    # 避免 temperature 為 0 導致除以 0 的錯誤，設定一個極小值
    t = max(temperature, 1e-6) 
    
    # 根據公式將 logits 除以溫控參數
    logits = logits / t
    
    # 數值穩定處理：減去最大值避免 exp 計算出 Inf
    logits_max = torch.max(logits, dim=-1, keepdim=True)[0]
    exp_logits = torch.exp(logits - logits_max)
    
    # 計算機率分佈
    probs = exp_logits / torch.sum(exp_logits, dim=-1, keepdim=True)
    
    return probs

class Sampler:
    
    @staticmethod
    def greedy( logits:torch.Tensor ):
        """
            input) logits.shape=(batch size, vocab size)
            output) next_token_id=(batch size, 1)
        """
        print( f"[INFO][SAMPLER] using GREEDY" )
        
        # 永遠選擇機率（或 logits）最大的 index
        # dim=-1 代表在 vocab size 那個維度找最大值
        next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
        
        return next_token_id

    @staticmethod
    def random( logits:torch.Tensor ):
        """
            input) logits.shape=(batch size, vocab size)
            output) next_token_id=(batch size, 1)
        """
        print( f"[INFO][SAMPLER] using RANDOM" )

        # 確保傳入的是機率值 (probs)
        logits_list = logits.squeeze(0).float().cpu().numpy() # 轉到 CPU 轉為 numpy 以利 random.choices
        token_id_list = np.arange( len(logits_list) )
        
        # 根據機率分佈進行隨機抽樣
        next_token_id = random.choices( token_id_list, weights=logits_list, k=1 )
        next_token_id = torch.tensor( next_token_id, device=device ).unsqueeze(0)

        return next_token_id

    @staticmethod
    def test( strategy:str, logits:torch.Tensor ):

        print( f"[INFO][SAMPLER][TEST] TESTING {strategy}" )

        # 為了觀察 LAB-4 的差異，我們連測 10 次
        results = []
        for i in range(10):
            match strategy:
                case "greedy":
                    token_id = Sampler.greedy( logits )
                    results.append(token_id.item())
                case "random":
                    token_id = Sampler.random( logits )
                    results.append(token_id.item())
                case _:
                    print( "[INFO][SAMPLER][TEST] No such strategy" )
                    return
        print(f"Results of 10 trials: {results}")
    
class Filter:
    # remember to change the logits to -inf if they are dropped

    @staticmethod
    def topK( probs:torch.Tensor, logits:torch.Tensor, threshold:float ):
        print( f"[INFO][FILTER] using TOP-K" )
        
        # threshold 在這裡就是 K
        k = int(threshold)
        
        # 找出前 K 大的值與索引
        top_values, _ = torch.topk(logits, k)
        
        # 取得第 K 個最大值（門檻值）
        min_top_value = top_values[:, -1].unsqueeze(-1)
        
        # 將小於門檻值的 logits 設為負無窮大
        logits[logits < min_top_value] = float('-inf')
        
        return logits
    
    @staticmethod
    def topP( probs:torch.Tensor, logits:torch.Tensor, threshold:float ):
        print( f"[INFO][FILTER] using TOP-P" )
        
        # 先對機率進行排序 (降序)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        
        # 計算累積機率
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # 找出累積機率超過門檻的位置
        # 我們要保留那些「累加後還沒超過 threshold」的，或是剛好超過的那一個
        sorted_indices_to_remove = cumulative_probs > threshold
        
        # 為了確保至少保留一個 Token，將第一個位置設為 False (不移除)
        # 並將遮罩向右移，保留剛好超過門檻的那個 token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        # 將要移除的 index 對應回原始 logits 並設為 -inf
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
        
        return logits
    
    @staticmethod
    def minP( probs:torch.Tensor, logits:torch.Tensor, threshold:float ):
        print( f"[INFO][FILTER] using MIN-P" )
        
        # 找出每一 batch 中的最大機率
        max_probs, _ = torch.max(probs, dim=-1, keepdim=True)
        
        # 計算實際的機率門檻
        limit = max_probs * threshold
        
        # 將機率低於 limit 的項在 logits 中設為 -inf
        logits[probs < limit] = float('-inf')
        
        return logits

    @staticmethod
    def test( strategy:str, probs:torch.Tensor, logits:torch.Tensor, threshold:float ):

        print( f"[INFO][FILTER][TEST] TESTING" )

        match strategy:
            case "topK":
                logits = Filter.topK( probs, logits, threshold )

            case "topP":
                logits = Filter.topP( probs, logits, threshold )

            case "minP":
                logits = Filter.minP( probs, logits, threshold )

            case _:
                print( "[INFO][FILTER][TEST] No such strategy" )

        print( logits )

class Penalty:
    
    @staticmethod
    def repetition( logits:torch.Tensor, seen:torch.Tensor, penalty:float ):
        """
        邏輯：對於出現過的 token，若 logit > 0 則除以 penalty，若 < 0 則乘以 penalty。
        """
        print( f"[INFO][PENALTY] using REPETITION" )

        # 建立一個遮罩，找出所有出現次數 > 0 的 index
        # seen 的形狀通常是 (vocab_size,)，需要擴展維度來匹配 (batch, vocab_size)
        mask = seen > 0
        
        # 套用公式：正值變小(除法)，負值更負(乘法)
        # 使用 torch.where 進行向量化運算
        logits[:, mask] = torch.where(
            logits[:, mask] > 0, 
            logits[:, mask] / penalty, 
            logits[:, mask] * penalty
        )

        return logits
    
    @staticmethod
    def frequency( logits:torch.Tensor, seen:torch.Tensor, penalty:float ):
        """
        邏輯：直接減去 (次數 * penalty)。出現次數愈多，分數扣愈重。
        """
        print( f"[INFO][PENALTY] using FREQUENCY" )

        # logits = logits - (count * penalty)
        logits = logits - (seen * penalty)

        return logits
    
    @staticmethod
    def presence( logits:torch.Tensor, seen:torch.Tensor, penalty:float ):
        """
        邏輯：只要出現過，就扣掉一次 penalty。不論出現 1 次還是 100 次，扣的分數都一樣。
        """
        print( f"[INFO][PENALTY] using PRESENCE" )

        # 只要次數 > 0，就視為「存在」，建立一個 0 或 1 的遮罩
        mask = (seen > 0).float()
        
        # logits = logits - (1 * penalty)
        logits = logits - (mask * penalty)

        return logits

    @staticmethod
    def test( strategy:str, logits:torch.Tensor, seen:torch.Tensor, penalty:float ):
        print( f"[INFO][PENALTY][TEST] TESTING {strategy} (penalty={penalty})" )

        match strategy:
            case "repetition":
                logits = Penalty.repetition( logits, seen, penalty )
            case "presence":
                logits = Penalty.presence( logits, seen, penalty )
            case "frequency":
                logits = Penalty.frequency( logits, seen, penalty )
            case _:
                print( "[INFO][PENALTY][TEST] No such strategy" )

        print( f"Result Logits: {logits}" )
        
class Tokenizer:

    @staticmethod
    def tokenize_pipe( tokenizer:AutoTokenizer, prompt:str ):
        """
            output) input_ids.shape=(batch size, token length)
            output) attention_masks.shape=(batch size, token length)
        """

        print( f"[INFO][TOKENIZE] using PIPE" )

        input_dict = tokenizer( prompt, return_tensors="pt", add_special_tokens=False )
        input_ids = input_dict[ "input_ids" ]
        attention_masks = input_dict[ "attention_mask" ]

        return input_ids, attention_masks

    @staticmethod
    def tokenize_step_by_step( tokenizer:AutoTokenizer, prompt:str ):
        """
            output) input_ids.shape=(batch size, token length)
            output) attention_masks.shape=(batch size, token length)
        """

        print( f"[INFO][TOKENIZE] using STEP-BY-STEP" )

        # [EXP] please implement here

        input_ids = torch.ones( 1, 1, dtype=torch.int64, device=device ) # just a temporary value
        attention_masks = torch.ones( 1, 1, dtype=torch.int64, device=device ) # just a temporary value

        return input_ids, attention_masks

    @staticmethod
    def test( strategy:str, tokenizer:AutoTokenizer, prompt:str ):

        print( f"[INFO][TOKENIZER][TEST] TESTING" )

        match strategy:
            case "pipe":
                input_ids, attention_masks = Tokenizer.tokenize_pipe( tokenizer, prompt )

            case "step-by-step":
                input_ids, attention_masks = Tokenizer.tokenize_step_by_step( tokenizer, prompt )

            case _:
                print( "[INFO][TOKENIZER][TEST] No such strategy" )

        print( input_ids )
        print( attention_masks )

class Generator:

    @staticmethod
    def generate_pipe( tokenizer:AutoTokenizer, model:AutoModelForCausalLM, input_ids:torch.Tensor, attention_masks:torch.Tensor, max_token_len=200 ):

        print( f"[INFO][GENERATE] using PIPE" )

        outputs = model.generate(
            input_ids, 
            attention_mask=attention_masks,
            max_new_tokens=max_token_len,  # avoid repeating without eos_token
            pad_token_id=tokenizer.eos_token_id
        )

        return outputs
    
    @staticmethod
    def generate_iterative( tokenizer:AutoTokenizer, model:AutoModelForCausalLM, input_ids:torch.Tensor, attention_masks:torch.Tensor, max_token_len=200, temperature=1.0):

        print( f"[INFO][GENERATE] using ITERATIVE" )

        seen = torch.zeros( len(tokenizer), device=device ) 
        for _ in range( max_token_len ): 

            outputs = model( input_ids )
            logits = outputs.logits 
            next_token_logits = logits[ :, -1, : ] 

            # --- [LAB-5] 套用 Penalty (以 Presence Penalty 為例) ---
            # 你可以手動切換不同策略來觀察效果
            next_token_logits = Penalty.presence(logits=next_token_logits, seen=seen, penalty=1.2)

            # --- [LAB-1/2] 計算機率 ---
            probs = Softmax(logits=next_token_logits, temperature=temperature)

            # --- [LAB-3] 套用 Filter (以 Top-P 為例) ---
            next_token_logits = Filter.topP(probs=probs, logits=next_token_logits, threshold=0.85)
            # 因為 Filter 修改了 logits，建議重新計算一次 probs 給 Sampler 使用
            probs = Softmax(logits=next_token_logits, temperature=temperature)

            # --- [LAB-4] 選擇 Sampler 策略 ---
            next_token_id = Sampler.random( logits=probs )
            next_token_id = Sampler.greedy( logits=next_token_logits )
            
            next_token = tokenizer.decode( next_token_id[0], skip_special_tokens=True )
            print( f"[INFO][DECODE ONE]>> {next_token}" )
            
            input_ids = torch.cat( [ input_ids, next_token_id ], dim=-1 ) 
            new_attn = torch.ones( 1, 1, device=device ) 
            attention_masks = torch.cat( [ attention_masks, new_attn ], dim=-1 )
            
            # --- [重要] 更新出現過的 token 次數，Penalty 才會生效 ---
            seen[next_token_id.item()] += 1
            
            if next_token_id.item() == tokenizer.eos_token_id:
                break

        return input_ids

def main():

    ### set user prompt ###

    user_prompt = "請列出三種奇怪的食物組合"
    
    ### load model and tokenizer ###

    print( f"[INFO] loading tokenizer & model" )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, dtype=torch.bfloat16)
    model.eval() # setting to evaluation mode -> freeze model weights

    ### set user prompt for chat format ###

    # chat models are tuned with chat format, so models need the format messages to know it needs to chat with users

    print( f"[INFO] converting user prompt to chat format" )

    prompt = [
        {"role": "user", "content": user_prompt }
    ]

    prompt = tokenizer.apply_chat_template(
                prompt, 
                tokenize=False, # not to tokenize because we want to tokenize ourselves
                add_generation_prompt=True # indicate model to start answering by adding assistant hint token
             )

    ### use tokenizer to get input_ids ( token id ) and attention mask ###

    print( f"[INFO] tokenizing" )

    input_ids, attention_masks = Tokenizer.tokenize_pipe( tokenizer=tokenizer, prompt=prompt )
    input_ids = input_ids.to(device) # token id sequence
    attention_masks = attention_masks.to(device) # used to mask padding part and future part

    ### generation ###

    print( f"[INFO] generating" )

    with torch.no_grad():
        output_ids = Generator.generate_iterative( tokenizer=tokenizer, model=model, attention_masks=attention_masks, input_ids=input_ids,temperature=0.1)

    ### decode generated input_ids to token sequence ###

    print( f"[INFO] decoding" )

    output = output_ids[0][len(input_ids[0]):] # only take answer part and specify 0 for first data because our batch size is 1
    output = tokenizer.decode(output, skip_special_tokens=True)

    ### print input & output for observasion ###

    print()
    print( f"[USER INPUT]: {user_prompt}" )
    print( f"[CHAT INPUT]: {prompt}" )
    print( f"[OUTPUT]: {output}" )

### device setting ###

device = torch.device( f"cuda:0" )
model_name = "/model"

if __name__ == '__main__':
    logits = torch.tensor([[3.0, 1.0, 0.0, -1.0, -2.0]])
    temperatures = [2.0, 1.0, 0.01] # T=0 會導致除以零，通常用極小值如 0.01 代替

    print("-" * 30)
    print("LAB-2: Logits Distribution Analysis")
    print(f"Original Logits: {logits}")
    print("-" * 30)

    for T in temperatures:
        probs = Softmax(logits=logits, temperature=T)
        # 格式化輸出
        formatted_probs = [f"{p.item():.4f}" for p in probs[0]]
        print(f"Temperature = {T:g}")
        print(f"Probabilities (A-E): {formatted_probs}")
        print("-" * 30)
    '''
    ### Example test code ###

    # test tokenizing ----------------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained( model_name )
    Tokenizer.test( strategy="step-by-step", tokenizer=tokenizer, prompt="你好" )

    # test softmax ----------------------------------------------------------------

    logits = torch.tensor( [[ 0.07, 0.06, 0.25, 0.41, 0.04, 0.03, 0.14 ]] )
    temperature = 1.0
    out_logits = Softmax( logits=logits, temperature=temperature )
    print( out_logits )
    
    # test filtering ----------------------------------------------------------------

    probs = torch.tensor( [[ 0.07, 0.06, 0.25, 0.41, 0.04, 0.03, 0.14 ]] )
    logits = torch.tensor( [[ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 ]] )
    threshold = 0.96
    Filter.test( strategy="topP", probs=probs, logits=logits, threshold=threshold )

    # test sampling ----------------------------------------------------------------

    logits = torch.tensor( [[ 0.07, 0.06, 0.25, 0.41, 0.04, 0.03, 0.14 ]] )
    Sampler.test( strategy="random", logits=logits )
    
    # test penalty ----------------------------------------------------------------

    logits = torch.tensor( [[ 2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0 ]] )
    seen = torch.tensor( [ 1, 0, 0, 2, 1, 0, 0 ] )
    penalty = 0.5
    Penalty.test( strategy="presence", logits=logits, seen=seen, penalty=penalty )
    '''
    
    # generating flow

    main()
