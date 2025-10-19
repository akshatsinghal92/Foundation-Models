import torch
import pickle

ckpt_path = "/Users/Patron/Desktop/Foundation-Models/Foundation-Models/nanoGPT/out-shakespeare-char-normal/ckpt.pt"
checkpoint = torch.load(ckpt_path, map_location='cpu')


# best_val_loss = checkpoint['best_val_loss']
# print(checkpoint.keys())
# print(checkpoint['model_args'])
print("best_specific_val_loss", checkpoint['best_specific_val_loss'])
print("best_generic_val_loss:", checkpoint['best_generic_val_loss'])


with open("/Users/Patron/Desktop/Foundation-Models/Foundation-Models/nanoGPT/data/harry_potter/meta.pkl", "rb") as f:
    a=pickle.load(f)

print(a.keys())

# python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=128 --batch_size=12 --n_layer=10 --n_head=10 --n_embd=200 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
