import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import pickle
import tiktoken
import re
import random
from model import GPT, GPTConfig

out_dir = '/Users/Patron/Desktop/Foundation-Models/out-harryp2-char'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1337
max_new_tokens = 64 
num_samples_per_prompt = 4 
num_updates = 50 
batch_size = 4 
lr = 1e-5
beta = 0.04 
clip_eps = 0.2 

save_path = "grpo_policy.pt"
checkpoint_dir = "grpo_checkpoints"

def verifier(text: str) -> float:
    count = text.lower().count('e')
    score_e = min(count, 20) / 20.0
    match = bool(re.match(r'^\s*Harry', text, re.IGNORECASE))
    score_regex = 1.0 if match else 0.0
    final_score = 0.5 * score_e + 0.5 * score_regex
    return final_score

torch.manual_seed(seed)
os.makedirs(checkpoint_dir, exist_ok=True)

def load_setup():
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
        
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
        
    return model, ref_model, encode, decode

def get_log_probs(model, input_ids, attention_mask):
    model.eval() 
    
    logits, _ = model(input_ids, targets=input_ids) 
    
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
    
    shift_mask = attention_mask[..., 1:].contiguous()
    token_log_probs = token_log_probs * shift_mask
    return token_log_probs

def grpo_step(model, ref_model, prompts, encode, decode, optimizer):
    model.train()
    
    all_inputs = []
    all_masks = []
    all_rewards = []
    prompt_lens = []
    
    total_loss = 0
    
    for prompt in prompts:
        p_ids = encode(prompt)
        p_tensor = torch.tensor([p_ids], dtype=torch.long, device=device)
        
        x = p_tensor.repeat(num_samples_per_prompt, 1)
        
        with torch.no_grad():
            y = model.generate(x, max_new_tokens, temperature=1.0, top_k=50)
            
        rewards = []
        texts = []
        for i in range(num_samples_per_prompt):
            text = decode(y[i].tolist())
            r = verifier(text)
            rewards.append(r)
            texts.append(text)
            
        rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
        
        mean_r = rewards.mean()
        std_r = rewards.std() + 1e-8
        advantages = (rewards - mean_r) / std_r
        
        mask = torch.ones_like(y, device=device)
        
        logits, _ = model(y, targets=y)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = y[..., 1:].contiguous()
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
        
        with torch.no_grad():
            ref_logits, _ = ref_model(y, targets=y)
            ref_shift_logits = ref_logits[..., :-1, :].contiguous()
            ref_log_probs = F.log_softmax(ref_shift_logits, dim=-1)
            ref_token_log_probs = torch.gather(ref_log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
            
        Lp = len(p_ids)
        
        gen_mask = torch.zeros_like(token_log_probs)
        gen_mask[:, Lp-1:] = 1.0
        
        kl_div = torch.exp(ref_token_log_probs) * (ref_token_log_probs - token_log_probs) 
        ratio_log = token_log_probs - ref_token_log_probs
        kl_penalty = beta * (torch.exp(ratio_log) - 1 - ratio_log) 
        
        adv_expanded = advantages.unsqueeze(1).expand_as(token_log_probs)
        
        selected_log_probs = token_log_probs * gen_mask
        selected_kl = (token_log_probs - ref_token_log_probs) * gen_mask 
        
        pg_loss = -(adv_expanded * selected_log_probs).sum(dim=1).mean()
        kl_loss = (selected_kl).sum(dim=1).mean()
        
        loss = pg_loss + beta * kl_loss
        
        loss.backward()
        total_loss += loss.item()
        
        all_rewards.extend(rewards.tolist())

    optimizer.step()
    optimizer.zero_grad()
    
    return total_loss / len(prompts), sum(all_rewards)/len(all_rewards)

def train():
    model, ref_model, encode, decode = load_setup()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    base_prompts = [
        "The", "Harry", "Hermione", "Ron", "It", "A", "In", "When", "If", "But",
        "He", "She", "They", "We", "I", "You", "There", "Here", "What", "Why",
        "Hogwarts", "Dumbledore", "Snape", "Malfoy", "Hagrid", "Sirius", "Voldemort"
    ]
    prompts_pool = base_prompts * 10
    
    print("Starting GRPO Training...")
    print(f"Steps: {num_updates}, Batch Size: {batch_size}, Group Size: {num_samples_per_prompt}")
    
    metrics = []
    
    for step in range(1, num_updates + 1):
        batch_prompts = random.sample(prompts_pool, batch_size)
        
        loss, mean_reward = grpo_step(model, ref_model, batch_prompts, encode, decode, optimizer)
        
        print(f"Step {step}/{num_updates} | Loss: {loss:.4f} | Mean Reward: {mean_reward:.4f}")
        metrics.append((step, loss, mean_reward))
        
        if step % 10 == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"step_{step}.pt"))
            
    torch.save(model.state_dict(), save_path)
    print(f"Saved GRPO model to {save_path}")
    return metrics

if __name__ == "__main__":
    train()
