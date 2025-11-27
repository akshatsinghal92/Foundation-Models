import torch
from model import GPT, GPTConfig
import torch.nn as nn
"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from reward_score import get_final_score




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


def score_text(text, reward_model, encode, device="cpu"):
    reward_model.eval()

    # Encode input → token IDs
    token_ids = encode(text)
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)   # (1, T)

    # No padding → attention_mask is all ones
    attention_mask = torch.ones_like(ids, dtype=torch.long)

    with torch.no_grad():
        score = reward_model(ids, attention_mask=attention_mask)

    return float(score.item())



# -----------------------
# Load reward model checkpoint
# -----------------------
ckpt = torch.load("/Users/Patron/Desktop/Foundation-Models/out-harryp2-char/full_reward_model.pt", map_location="cpu")


# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = '/Users/Patron/Desktop/Foundation-Models/out-harryp2-char' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 15 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1099
device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------




torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)





ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

print("Base GPT loaded!")


# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)


start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

sample_texts=[]
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            sample_texts.append(decode(y[0].tolist()))
            # print("score: ", score_text(decode(y[0].tolist()),reward_model, encode, device))
            print('---------------')



reward_head = RewardHead(n_embd=checkpoint['model_args']["n_embd"])
reward_head.load_state_dict(ckpt["reward_head_state_dict"])
reward_head.eval()

print("Reward head loaded!")


reward_model = RewardWrapper(model, reward_head)
reward_model.eval()

print("Reward model ready!")


max_len=128 
for text in sample_texts:
    # y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    # print(decode(y[0].tolist()))
    # sample_texts.append(decode(y[0].tolist()))
    if len(text) > max_len:
        text = text[:max_len]
    print(text)
    print("score: ", score_text(text,reward_model, encode, device))
    print('---------------')




# text=sample_texts[0]
# print(text)

# device = "cpu"  # or "cuda" if available

# # text = "Harry looked at Ron and smiled."

# text = "Hizary looked at Ron and smildknd."

# score = score_text(text, reward_model, encode, device)
# print("Reward score:", score)

