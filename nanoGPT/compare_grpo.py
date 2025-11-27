import os
import torch
import pickle
import tiktoken
import re
import random
from model import GPT, GPTConfig


out_dir = '/Users/Patron/Desktop/Foundation-Models/out-harryp2-char'
grpo_path = 'grpo_policy.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1337
max_new_tokens = 64
num_prompts = 50
temperature = 1.0
top_k = 50


def verifier(text: str) -> float:
    count = text.lower().count('e')
    score_e = min(count, 20) / 20.0
    match = bool(re.match(r'^\s*Harry', text, re.IGNORECASE))
    score_regex = 1.0 if match else 0.0
    final_score = 0.5 * score_e + 0.5 * score_regex
    return final_score

def load_model(path, is_state_dict=False):
    print(f"Loading model from {path}...")
    checkpoint = torch.load(path, map_location=device)
    
    if is_state_dict:
        
        base_ckpt = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location=device)
        gptconf = GPTConfig(**base_ckpt['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint
    else:
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']

    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model

def get_tokenizer():
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    load_meta = False
    if 'config' in checkpoint and 'dataset' in checkpoint['config']:
        meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
        load_meta = os.path.exists(meta_path)
    
    if load_meta:
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
    return encode, decode

def evaluate(model, prompts, encode, decode, name="Model"):
    print(f"\nEvaluating {name}...")
    total_score = 0
    samples = []
    
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            start_ids = encode(prompt)
            x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
            
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            text = decode(y[0].tolist())
            
            score = verifier(text)
            total_score += score
            samples.append((score, text, prompt))
            
    avg_score = total_score / len(prompts)
    return avg_score, samples

def main():
    torch.manual_seed(seed)
    encode, decode = get_tokenizer()
    
    # Prepare Prompts
    base_prompts = [
        "The", "Harry", "Hermione", "Ron", "It", "A", "In", "When", "If", "But",
        "He", "She", "They", "We", "I", "You", "There", "Here", "What", "Why",
        "Hogwarts", "Dumbledore", "Snape", "Malfoy", "Hagrid", "Sirius", "Voldemort"
    ]
    prompts = base_prompts * 2
    random.shuffle(prompts)
    prompts = prompts[:num_prompts]
    
    # 1. Evaluate Base Model
    base_model = load_model(os.path.join(out_dir, 'ckpt.pt'), is_state_dict=False)
    base_score, base_samples = evaluate(base_model, prompts, encode, decode, name="Base Model")
    del base_model # Free memory
    
    # 2. Evaluate GRPO Model
    grpo_model = load_model(grpo_path, is_state_dict=True)
    grpo_score, grpo_samples = evaluate(grpo_model, prompts, encode, decode, name="GRPO Model")
    
    # 3. Report Results
    print("\n" + "="*40)
    print("COMPARISON RESULTS")
    print("="*40)
    print(f"Base Model Mean Score: {base_score:.4f}")
    print(f"GRPO Model Mean Score: {grpo_score:.4f}")
    print(f"Improvement:           {grpo_score - base_score:+.4f}")
    
    print("\n" + "-"*40)
    print("Qualitative Comparison (Same Prompts)")
    print("-"*40)
    
    # Show 5 random comparisons
    indices = random.sample(range(len(prompts)), 5)
    for idx in indices:
        b_score, b_text, prompt = base_samples[idx]
        g_score, g_text, _ = grpo_samples[idx]
        
        print(f"\nPrompt: '{prompt}'\\\\")
        print(f"Base (Score: {b_score:.2f}): {b_text}\\\\")
        print(f"GRPO (Score: {g_score:.2f}): {g_text}\\\\")

if __name__ == "__main__":
    main()
