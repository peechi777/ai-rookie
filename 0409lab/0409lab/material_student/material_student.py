import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import numpy as np

def Softmax( logits:torch.Tensor, temperature:float=1.0 ):
    """
        input) logits.shape=(batch size, vocab size)
        output) probs.shape=(batch size, vocab size)
    """

    print( f"[INFO][SOFTMAX] using SOFTMAX" )

    # [EXP] please implement here

    # [hint] asign minumum value to avoid temperature=0 that arises divided by 0 error
    # [hint] if you implement by exp functions, please minus the max logit makes the max value to 1 and others become smaller values 
    # to avoid infinite large value of exponential

    probs = torch.ones_like( logits, dtype=torch.int64, device=device ) # just a temporary value
    
    return probs

class Sampler:
    
    @staticmethod
    def random( logits:torch.Tensor ):
        """
            input) logits.shape=(batch size, vocab size)
            output) next_token_id=(batch size, 1)
        """

        print( f"[INFO][SAMPLER] using RANDOM" )

        logits_list = logits.squeeze(0)
        token_id_list = np.arange( len(logits_list) )
        next_token_id = random.choices( token_id_list, weights=logits_list, k=1 )
        next_token_id = torch.tensor( next_token_id, device=device ).unsqueeze(0)

        return next_token_id

    @staticmethod
    def test( strategy:str, logits:torch.Tensor ):

        print( f"[INFO][SAMPLER][TEST] TESTING" )

        match strategy:
            case "random":
                token_id = Sampler.random( logits )

            case _:
                print( "[INFO][SAMPLER][TEST] No such strategy" )

        print( token_id )

class Filter:
    # remember to change the logits to -inf if they are dropped

    @staticmethod
    def topK( probs:torch.Tensor, logits:torch.Tensor, threshold:float ):
        """
            input) probs.shape=(batch size, vocab size)
            input) logits.shape=(batch size, vocab size)
            output) logits.shape=(batch size, vocab size)
        """

        print( f"[INFO][FILTER] using TOP-K" )

        # [EXP] please implement here

        return logits
    
    @staticmethod
    def topP( probs:torch.Tensor, logits:torch.Tensor, threshold:float ):
        """
            input) probs.shape=(batch size, vocab size)
            input) logits.shape=(batch size, vocab size)
            output) logits.shape=(batch size, vocab size)
        """

        print( f"[INFO][FILTER] using TOP-P" )

        # [EXP] please implement here

        return logits
    
    @staticmethod
    def minP( probs:torch.Tensor, logits:torch.Tensor, threshold:float ):
        """
            input) probs.shape=(batch size, vocab size)
            input) logits.shape=(batch size, vocab size)
            output) logits.shape=(batch size, vocab size)
        """

        print( f"[INFO][FILTER] using MIN-P" )

        # [EXP] please implement here

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
            input) logits.shape=(batch size, vocab size)
            output) logits.shape=(batch size, vocab size)
        """

        print( f"[INFO][PENALTY] using REPETITION" )

        # [EXP] please implement here

        return logits
    
    @staticmethod
    def frequency( logits:torch.Tensor, seen:torch.Tensor, penalty:float ):
        """
            input) logits.shape=(batch size, vocab size)
            output) logits.shape=(batch size, vocab size)
        """

        print( f"[INFO][PENALTY] using FREQUENCY" )

        # [EXP] please implement here

        return logits
    
    @staticmethod
    def presence( logits:torch.Tensor, seen:torch.Tensor, penalty:float ):
        """
            input) logits.shape=(batch size, vocab size)
            output) logits.shape=(batch size, vocab size)
        """

        print( f"[INFO][PENALTY] using PRESENCE" )

        # [EXP] please implement here

        return logits

    @staticmethod
    def test( strategy:str, logits:torch.Tensor, seen:torch.Tensor, penalty:float ):

        print( f"[INFO][PENALTY][TEST] TESTING" )

        match strategy:
            case "repetition":
                logits = Penalty.repetition( logits, seen, penalty )

            case "presence":
                logits = Penalty.presence( logits, seen, penalty )

            case "frequency":
                logits = Penalty.frequency( logits, seen, penalty )

            case _:
                print( "[INFO][PENALTY][TEST] No such strategy" )

        print( logits )

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
    def generate_iterative( tokenizer:AutoTokenizer, model:AutoModelForCausalLM, input_ids:torch.Tensor, attention_masks:torch.Tensor, max_token_len=200 ):

        print( f"[INFO][GENERATE] using ITERATIVE" )

        seen = torch.zeros( len(tokenizer), device=device ) # record the appear count of tokens -> seen[ token_id ] = appear_count
        for _ in range( max_token_len ): # avoid repeating without eos_token

            ### generate by model ###

            outputs = model( input_ids )
            
            ### obtain logits ###
            
            logits = outputs.logits # get generated logits
            # logits.shape: [batch size, suquence length, vocab size]
            # - sequence length: token number of input, model will predict for all positions
            # - vocab size: size of vocabulary
            next_token_logits = logits[ :, -1, : ] # only get the last position because we only want to generate the last token

            ### (optional) apply penalty ###

            # [EXP] please implement here

            ### (optional) topP, topK, minP ###
            
            # [EXP] please implement here
            
            ### use softmax to convert logits to probabilities ###

            # [EXP] please implement here
            
            ### choose the next token ###
            
            next_token_id = Sampler.random( logits=next_token_logits )
            
            ### decode the token id to text ###

            next_token = tokenizer.decode( next_token_id[0], skip_special_tokens=True )
            print( f"[INFO][DECODE ONE]>> {next_token}" )
            
            ### update the input_ids by concat the new token to previous token sequence ###

            input_ids = torch.cat( [ input_ids, next_token_id ], dim=-1 ) # concat generated token with previous sequence for next iteration
            new_attn = torch.ones( 1, 1, device=device ) # new mask=1 for generated token because it is a valid token, not padding
            attention_masks = torch.cat( [ attention_masks, new_attn ], dim=-1 )
            
            ### update the appeared list ###

            # [EXP] please implement here
            
            ### stop generating if end-of-sequence token is generated ###

            if next_token_id.item() == tokenizer.eos_token_id:
                break

        ### return generated result ###

        return input_ids

def main():

    ### set user prompt ###

    user_prompt = "你喜歡吃什麼食物?"
    
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
        output_ids = Generator.generate_iterative( tokenizer=tokenizer, model=model, attention_masks=attention_masks, input_ids=input_ids )

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
