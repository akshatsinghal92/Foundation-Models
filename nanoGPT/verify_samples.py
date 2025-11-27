"""
Generate samples from the base model and run the verifier on them.
"""
import os
import torch
import pickle
import tiktoken
import re
from model import GPT, GPTConfig

out_dir = '/Users/Patron/Desktop/Foundation-Models/out-harryp2-char'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1337
max_new_tokens = 200
num_samples = 10
temperature = 1.0
top_k = 50

def verifier(text: str) -> float:
    count = text.lower().count('e')
    score_e = min(count, 20) / 20.0
    match = bool(re.match(r'^\s*Harry', text, re.IGNORECASE))
    score_regex = 1.0 if match else 0.0
    final_score = 0.5 * score_e + 0.5 * score_regex
    return final_score

# Load Model
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
print(f"Loading model from {ckpt_path}...")
checkpoint = torch.load(ckpt_path, map_location=device)
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

# Load Tokenizer
load_meta = False
if 'config' in checkpoint and 'dataset' in checkpoint['config']:
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)

if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# Generate and Verify
prompts = ["The", "Harry", "Hermione", "Ron", "It"]
prompts = prompts * (num_samples // len(prompts) + 1)
prompts = prompts[:num_samples]

print(f"\nGenerating {num_samples} samples and verifying...\n")

total_score = 0
for i, prompt in enumerate(prompts):
    start_ids = encode(prompt)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    
    with torch.no_grad():
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        text = decode(y[0].tolist())
        
    score = verifier(text)
    total_score += score
    
    print(f"Sample {i+1} (Prompt: '{prompt}'):")
    print(f"Score: {score:.4f}")
    print(f"Text: {text[:100]}...")
    print("-" * 20)

print(f"\nAverage Score: {total_score / num_samples:.4f}")
