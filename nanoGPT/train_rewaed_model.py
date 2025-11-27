import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from reward_score import get_final_score
from torch.utils.data import random_split



class RewardDataset(Dataset):
    def __init__(self, texts, encode_fn, scorer, max_len=None):
        self.items = []
        for t in texts:
            toks = encode_fn(t)
            if max_len is not None and len(toks) > max_len:
                toks = toks[:max_len]
            reward = scorer(t)
            self.items.append((torch.tensor(toks, dtype=torch.long), torch.tensor(float(reward), dtype=torch.float)))
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        return self.items[idx]

def collate(batch, device=torch.device("cpu")):
    seqs, rewards = zip(*batch)
    input_ids = torch.stack(seqs).to(device)
    attn = (input_ids != 0).long().to(device)  # infer mask if pad_id==0
    rewards = torch.stack(rewards).to(torch.float).to(device)
    return {"input_ids": input_ids, "attention_mask": attn, "rewards": rewards}

class RewardHead(nn.Module):
    def __init__(self, n_embd: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, h):
        return self.net(h).squeeze(-1)

class RewardWrapper(nn.Module):
   
    def __init__(self, base_model, reward_head):
        super().__init__()
        self.base = base_model
        self.reward_head = reward_head
        if hasattr(self.base, "lm_head"):
            try:
                self.base.lm_head = nn.Identity()
            except Exception:
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

        if hasattr(transformer, "h"):
            for block in transformer.h:
                x = block(x)
        else:
            out = self.base(input_ids)
            raise RuntimeError("Transformer blocks missing; fallback not implemented.")

        if hasattr(transformer, "ln_f"):
            x = transformer.ln_f(x)

        
        if attention_mask is None:
            final = x[:, -1, :]
        else:
            lengths = attention_mask.sum(dim=1)
            lengths = torch.clamp(lengths, min=1)
            idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, x.size(2))
            final = x.gather(1, idx).squeeze(1)

        return self.reward_head(final)
def train_reward(reward_model, dataset, device, save_path,
                 epochs=3, batch_size=8, lr=1e-4):
    """
    Train reward_model on `dataset`, use 90/10 train/val split, save best full model
    to f"{save_path}/full_reward_model.pt" based on validation loss.
    """

    params = [p for p in reward_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    criterion = nn.MSELoss()

    # train/val split (90/10)
    n = len(dataset)
    if n == 0:
        raise ValueError("Empty dataset")
    train_size = int(0.9 * n)
    val_size = n - train_size
    if val_size <= 0:
        
        train_ds = dataset
        val_ds = None
    else:
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

    try:
        collate_fn = collate
    except NameError:
        collate_fn = None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn) if val_size > 0 else None

    reward_model.to(device)
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for ep in range(1, epochs + 1):
       
        reward_model.train()
        total_loss = 0.0
        steps = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            rewards = batch["rewards"].to(device)

            preds = reward_model(input_ids=input_ids, attention_mask=attn)
            
            if preds.dim() == 2 and preds.size(1) == 1:
                preds = preds.squeeze(-1)

            loss = criterion(preds, rewards)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            total_loss += loss.item()
            steps += 1
            train_losses.append(loss.item())

        avg_train_loss = total_loss / steps if steps else 0.0

       
        avg_val_loss = None
        if val_loader is not None:
            reward_model.eval()
            val_total = 0.0
            val_steps = 0
            with torch.no_grad():
                for vb in val_loader:
                    v_input_ids = vb["input_ids"].to(device)
                    v_attn = vb["attention_mask"].to(device)
                    v_rewards = vb["rewards"].to(device)

                    v_preds = reward_model(input_ids=v_input_ids, attention_mask=v_attn)
                    if v_preds.dim() == 2 and v_preds.size(1) == 1:
                        v_preds = v_preds.squeeze(-1)
                    v_loss = criterion(v_preds, v_rewards)

                    val_total += v_loss.item()
                    val_steps += 1
                    val_losses.append(v_loss.item())

            avg_val_loss = val_total / val_steps if val_steps else 0.0

            # save only if validation improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_data = {
                    "base_model_state_dict": reward_model.base.state_dict(),
                    "reward_head_state_dict": reward_model.reward_head.state_dict(),
                    "best_loss": best_val_loss,
                    "epoch": ep
                }
                os.makedirs(save_path, exist_ok=True)
                torch.save(save_data, f"{save_path}/full_reward_model.pt")
                print(f"Epoch {ep}: New best val loss {best_val_loss:.4f} → saved {save_path}/full_reward_model.pt")

        else:
            
            if avg_train_loss < best_val_loss:
                best_val_loss = avg_train_loss
                save_data = {
                    "base_model_state_dict": reward_model.base.state_dict(),
                    "reward_head_state_dict": reward_model.reward_head.state_dict(),
                    "best_loss": best_val_loss,
                    "epoch": ep
                }
                os.makedirs(save_path, exist_ok=True)
                torch.save(save_data, f"{save_path}/full_reward_model.pt")
                print(f"Epoch {ep}: New best train loss {best_val_loss:.4f} → saved {save_path}/full_reward_model.pt")

        print(f"Epoch {ep} | train_loss = {avg_train_loss:.4f}" + (f" | val_loss = {avg_val_loss:.4f}" if avg_val_loss is not None else ""))

    return {"train_losses": train_losses, "val_losses": val_losses, "best_loss": best_val_loss}


seed = 1337
device = 'cpu'

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


ckpt_path = os.path.join('/Users/Patron/Desktop/Foundation-Models/out-harryp2-char/ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

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
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)


hidden_size = gptconf.n_embd
print(hidden_size)
reward_head = RewardHead(hidden_size, hidden=100)
reward_model = RewardWrapper(model, reward_head).to(device)

for p in reward_model.reward_head.parameters():
    p.requires_grad = True


save_path="/Users/Patron/Desktop/Foundation-Models/out-harryp2-char"
# torch.save({"texts_generated": sample_texts}, f"{save_path}/generated_text.pt")
sample_texts=torch.load("/Users/Patron/Desktop/Foundation-Models/out-harryp2-char/generated_text.pt")['texts_generated']
scorer = lambda s: get_final_score(s)
dataset = RewardDataset(sample_texts, encode, scorer, max_len=128)
train_reward(reward_model, dataset, device=device, save_path=save_path, epochs=50, batch_size=16, lr=1e-3)


