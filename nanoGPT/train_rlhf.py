import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Tuple
import copy
import pickle
from contextlib import nullcontext
import tiktoken
from model import GPT, GPTConfig
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROMPTS = ["Harry looked at Ron and said,", "The castle was quiet as", "Hermione whispered,"]
num_episodes_per_prompt = 10
max_new_tokens = 80
temperature = 0.9
top_k = 50
num_updates = 20
batch_size = 8
lr = 1e-5
reward_scale = 1.0
baseline_decay = 0.99
fine_tune_base = True
save_aligned_path = "aligned_policy.pt"
checkpoint_dir = "rlhf_checkpoints"
eval_interval = 5
seed = 1337
beta = 0.5

torch.manual_seed(seed)
os.makedirs(checkpoint_dir, exist_ok=True)

class RewardHead(nn.Module):
    def __init__(self, n_embd: int, hidden: int = 100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, h):
        return self.net(h).squeeze(-1)

class RewardWrapper(nn.Module):
    def __init__(self, base, reward_head):
        super().__init__()
        self.base = base
        self.reward_head = reward_head
        try:
            self.base.lm_head = nn.Identity()
        except:
            pass

    def forward(self, input_ids, attention_mask=None):
        transformer = self.base.transformer
        device = input_ids.device

        tok_emb = transformer.wte(input_ids)
        if hasattr(transformer, "wpe"):
            pos = torch.arange(0, input_ids.size(1), device=device).unsqueeze(0)
            pos_emb = transformer.wpe(pos)
            x = tok_emb + pos_emb
        else:
            x = tok_emb

        if hasattr(transformer, "drop"):
            x = transformer.drop(x)

        for block in transformer.h:
            x = block(x)

        if hasattr(transformer, "ln_f"):
            x = transformer.ln_f(x)

        if attention_mask is None:
            final_hidden = x[:, -1, :]
        else:
            lengths = attention_mask.sum(dim=1)
            lengths = torch.clamp(lengths, min=1)
            idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, x.size(2))
            final_hidden = x.gather(1, idx).squeeze(1)

        return self.reward_head(final_hidden)

def sample_from_policy(policy_model, prompt_ids: List[int], num_samples=1, device=device):
    policy_model.eval()
    policy_model.to(device)
    out_ids = []
    max_len = 128

    with torch.no_grad():
        x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        if x.shape[1] > max_len:
            x = x[:, :max_len]
        for _ in range(num_samples):
            y = policy_model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
            out_ids.append(y[0].tolist())
    return out_ids

def score_sequences(reward_model, seqs: List[List[int]]):
    max_len = 128
    input_ids = torch.full((len(seqs), max_len), 0, dtype=torch.long, device=device)
    attn = torch.zeros((len(seqs), max_len), dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        l = len(s)
        if l > max_len:
            s = s[:max_len]
        input_ids[i, :l] = torch.tensor(s, dtype=torch.long, device=device)
        attn[i, :l] = 1
    reward_model.eval()
    with torch.no_grad():
        scores = reward_model(input_ids=input_ids, attention_mask=attn)
    return scores.detach()

def get_log_probs(model, input_ids, attention_mask):
    model.eval()
    with torch.no_grad():
        logits, _ = model(input_ids, targets=input_ids)
    
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
    
    shift_mask = attention_mask[..., 1:].contiguous()
    token_log_probs = token_log_probs * shift_mask
    
    return token_log_probs

def reinforce_update(policy_model, ref_model, reward_model, prompts_ids: List[List[int]], optimizer, baseline, pad_id=0):
    policy_model.train()
    ref_model.eval()

    all_seqs = []
    prompt_lens = []
    for p in prompts_ids:
        for _ in range(num_episodes_per_prompt):
            seqs = sample_from_policy(policy_model, p, num_samples=1, device=device)
            full_ids = seqs[0]
            all_seqs.append(full_ids)
            prompt_lens.append(len(p))

    B_total = len(all_seqs)
    
    max_len = max(len(s) for s in all_seqs)
    block_size = policy_model.config.block_size
    max_len = min(max_len, block_size)

    ids = torch.full((B_total, max_len), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((B_total, max_len), dtype=torch.long, device=device)

    for i, s in enumerate(all_seqs):
        L = len(s)
        if L > max_len:
            s = s[-max_len:]
            L = max_len
        ids[i, :L] = torch.tensor(s, dtype=torch.long, device=device)
        mask[i, :L] = 1

    raw_rewards = score_sequences(reward_model, all_seqs).to(device) * reward_scale
    
    with torch.no_grad():
        ref_log_probs = get_log_probs(ref_model, ids, mask)
        policy_log_probs_detached = get_log_probs(policy_model, ids, mask)
    
    kl_per_token = policy_log_probs_detached - ref_log_probs
    
    kl_rewards = []
    for i, Lp in enumerate(prompt_lens):
        seq_len = mask[i].sum().item()
        start_idx = Lp - 1
        end_idx = seq_len - 1
        
        if end_idx > start_idx:
            seq_kl = kl_per_token[i, start_idx:end_idx].sum()
        else:
            seq_kl = torch.tensor(0.0, device=device)
        kl_rewards.append(seq_kl)
    
    kl_rewards = torch.stack(kl_rewards)
    total_rewards = raw_rewards - beta * kl_rewards
    
    baseline_val = baseline["value"]
    baseline["value"] = baseline_decay * baseline["value"] + (1 - baseline_decay) * total_rewards.mean().item()

    logits, _ = policy_model(ids, targets=ids)
    logp = F.log_softmax(logits, dim=-1)
    per_token_logp = torch.gather(logp, dim=2, index=ids.unsqueeze(-1)).squeeze(-1)
    
    seq_logprob = []
    for i, Lp in enumerate(prompt_lens):
        seq_len = mask[i].sum().item()
        actual_len = len(all_seqs[i])

        if actual_len > max_len:
            tokens_removed = actual_len - max_len
            adjusted_Lp = max(0, Lp - tokens_removed)
        else:
            adjusted_Lp = Lp

        gen_len = seq_len - adjusted_Lp
        if gen_len <= 0:
            seq_logprob.append(torch.tensor(0.0, device=device))
        else:
            gen_logp = per_token_logp[i, adjusted_Lp: adjusted_Lp + gen_len].sum()
            seq_logprob.append(gen_logp)
    seq_logprob = torch.stack(seq_logprob)

    adv = (total_rewards - baseline_val).detach()
    loss = -(adv * seq_logprob).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in policy_model.parameters() if p.requires_grad], 1.0)
    optimizer.step()

    return {
        "loss": loss.item(),
        "reward_mean": total_rewards.mean().item(),
        "raw_reward_mean": raw_rewards.mean().item(),
        "kl_mean": kl_rewards.mean().item(),
        "adv_mean": adv.mean().item()
    }

def run_reinforce_training(policy_model, reward_model, encode, decode):
    print("Creating reference model...")
    ref_model = copy.deepcopy(policy_model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    
    if fine_tune_base:
        for p in policy_model.parameters():
            p.requires_grad = True
    else:
        for p in policy_model.parameters():
            p.requires_grad = False
        if hasattr(policy_model, "lm_head"):
            for p in policy_model.lm_head.parameters():
                p.requires_grad = True

    trainable = [p for p in policy_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    baseline = {"value": 0.0}

    print("=== Evaluation BEFORE RL fine-tuning ===")
    before_samples = []
    before_scores = []
    for prompt in PROMPTS:
        p_ids = encode(prompt)
        seqs = sample_from_policy(policy_model, p_ids, num_samples=4, device=device)
        for s in seqs:
            score = score_sequences(reward_model, [s])[0].item()
            before_samples.append((score, decode(s)))
            before_scores.append(score)
    before_avg = sum(before_scores)/len(before_scores)
    print(f"Before avg reward: {before_avg:.4f}")
    before_samples.sort(reverse=True)
    print("Top before sample:")
    print(before_samples[0][1])
    print("Score:", before_samples[0][0])

    prompts_ids = [encode(p) for p in PROMPTS]

    for update in range(1, num_updates+1):
        batch_prompts = []
        for _ in range(batch_size):
            batch_prompts.append(random.choice(prompts_ids))
        
        stats = reinforce_update(policy_model, ref_model, reward_model, batch_prompts, optimizer, baseline)
        
        if update % eval_interval == 0 or update == 1:
            print(f"[Update {update}] loss={stats['loss']:.4f} reward={stats['reward_mean']:.4f} (raw={stats['raw_reward_mean']:.4f}, kl={stats['kl_mean']:.4f})")
        if update % (eval_interval*5) == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"update{update}.pt")
            torch.save(policy_model.state_dict(), ckpt_path)

    print("=== Evaluation AFTER RL fine-tuning ===")
    after_samples = []
    after_scores = []
    for prompt in PROMPTS:
        p_ids = encode(prompt)
        seqs = sample_from_policy(policy_model, p_ids, num_samples=8)
        for s in seqs:
            score = score_sequences(reward_model, [s])[0].item()
            after_samples.append((score, decode(s)))
            after_scores.append(score)
    after_avg = sum(after_scores)/len(after_scores)
    print(f"After avg reward: {after_avg:.4f}")
    after_samples.sort(reverse=True)
    print("Top after sample:")
    print(after_samples[0][1])
    print("Score:", after_samples[0][0])

    torch.save(policy_model.state_dict(), save_aligned_path)
    print("Saved aligned policy to", save_aligned_path)

    improvement = after_avg - before_avg
    print(f"\n=== RLHF Training Summary ===")
    print(f"Before avg reward: {before_avg:.4f}")
    print(f"After avg reward:  {after_avg:.4f}")
    print(f"Improvement:       {improvement:+.4f} ({improvement/abs(before_avg)*100:+.1f}%)")

    return {
        "before_avg": before_avg,
        "after_avg": after_avg,
        "improvement": improvement,
        "before_top": before_samples[0],
        "after_top": after_samples[0]
    }

out_dir = '/Users/Patron/Desktop/Foundation-Models/out-harryp2-char'
init_from = 'resume'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False

torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

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
model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

print("Base GPT loaded!")
policy_model = copy.deepcopy(model)

load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
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

reward_ckpt = torch.load(os.path.join(out_dir, "full_reward_model.pt"), map_location="cpu")
reward_head = RewardHead(n_embd=checkpoint['model_args']["n_embd"])
reward_head.load_state_dict(reward_ckpt["reward_head_state_dict"])
reward_head.eval()

print("Reward head loaded!")

reward_model = RewardWrapper(model, reward_head)
reward_model.eval()

print("Reward model ready!")

result = run_reinforce_training(policy_model, reward_model, encode, decode)
print(result)
