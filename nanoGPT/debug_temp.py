import os
import pickle
import requests
import numpy as np



input_file_path = "/Users/Patron/Desktop/Foundation-Models/Foundation-Models/nanoGPT/data/shakespeare_char/input.txt"
with open(input_file_path, 'r') as f:
    data = f.read()


print(f"length of dataset in characters: {len(data):,}")

chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")



input_file_path ="/Users/Patron/Desktop/Foundation-Models/Foundation-Models/nanoGPT/data/harry_potter/04 Harry Potter and the Goblet of Fire.txt"

with open(input_file_path, 'r') as f:
    data1 = f.read()


chars1 = sorted(list(set(data1)))
vocab_size1 = len(chars1)
print("all the unique characters:", ''.join(chars1))
print(f"vocab size: {vocab_size1:,}")

extra_chars=""
for i in chars1:
    if i not in chars:
        extra_chars+=i

for i in extra_chars:
    data1=data1.replace(i, "")

chars1 = sorted(list(set(data1)))
vocab_size1 = len(chars1)
print("all the unique characters:", ''.join(chars1))
print(f"vocab size: {vocab_size1:,}")