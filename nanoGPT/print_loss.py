import torch
import pickle

ckpt_path = "/Users/Patron/Desktop/Foundation-Models1/Foundation-Models/nanoGPT/out-harryp2-char/ckpt.pt"
checkpoint = torch.load(ckpt_path, map_location='cpu')


print("best_specific_val_loss", checkpoint['best_specific_val_loss'])
print("best_generic_val_loss:", checkpoint['best_generic_val_loss'])


