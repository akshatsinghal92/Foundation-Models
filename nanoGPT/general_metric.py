import torch
import re
from rapidfuzz import process, fuzz
from nltk.corpus import words


import nltk
from collections import Counter

nltk.download('brown')
nltk.download('universal_tagset')

from nltk.corpus import brown

all_words = [w.lower() for w in brown.words() if w.isalpha()]

freq = Counter(all_words)

N = 5000
vocab = set([w for w, _ in freq.most_common(N)])


def logits_to_text(logits, itos):

    pred_indices = torch.argmax(logits, dim=-1)  
    
    batch_text = []
    for seq in pred_indices:
        text = ''.join([itos[i.item()] for i in seq])
        batch_text.append(text)
    return batch_text


def text_to_words(batch_text):
    batch_words = []
    for text in batch_text:
        words_in_text = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        batch_words.append(words_in_text)
    return batch_words




def fuzzy_word_metric(batch_words, vocab):

    total_mismatch = 0
    total_words = 0
    for word_list in batch_words:
        for w in word_list:
            # print(w)
            closest_word, score, _ = process.extractOne(w, vocab, scorer=fuzz.ratio)
            mismatch_chars = len(w) - int(score / 100 * len(closest_word))
            total_mismatch += mismatch_chars
            total_words += 1
            
                
    avg_mismatch = total_mismatch / max(1, total_words)
    
    
    return avg_mismatch

def get_score(logits, itos):
    # print("General metric start")
    batch_text = logits_to_text(logits, itos)
    # print("General metric start")
    batch_words = text_to_words(batch_text)
    # print("General metric start")
    avg_mismatch = fuzzy_word_metric(batch_words, vocab)
    # print(avg_mismatch)
    return avg_mismatch

