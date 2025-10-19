import torch
import pickle

ckpt_path = "/Users/Patron/Desktop/Foundation-Models/Foundation-Models/nanoGPT/out-shakespeare-char-normal/ckpt.pt"
checkpoint = torch.load(ckpt_path, map_location='cpu')


print("best_specific_val_loss", checkpoint['best_specific_val_loss'])
print("best_generic_val_loss:", checkpoint['best_generic_val_loss'])


with open("/Users/Patron/Desktop/Foundation-Models/Foundation-Models/nanoGPT/data/harry_potter/meta.pkl", "rb") as f:
    a=pickle.load(f)

print(a.keys())
