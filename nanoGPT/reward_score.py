

import re
from typing import List, Dict, Set, Optional
from collections import Counter
from wordfreq import top_n_list
from rapidfuzz import fuzz, process


fiction_names = [
    "Harry", "Ron", "Hermione", "Dumbledore", "Snape", "Hagrid",
    "McGonagall", "Moody", "Karkaroff", "Dobby", "Winky",
    "Krum", "Crouch", "Bagman", "Voldemort", "Weasley", "Malfoy",
    "Fred", "George", "Percy", "Neville", "Luna", "Sirius"
]


fiction_names = {n.lower() for n in fiction_names}

english_words=top_n_list("en", 50000)
english_words={w.lower() for w in english_words}.union(fiction_names)


def word_match_score(word, name_list):
    if not word:
        return 0.0
    
    name_list=[name.lower() for name in name_list]
    match = process.extractOne(word, name_list, scorer=fuzz.ratio)
    if match is None:
        return 0.0
    _, score, _ = match
    return float(score)

def word_is_english(word):

    w = word.lower()
    if not w.isalpha():
        return False
    if w in english_words:
        return True
    if len(w) <= 2:
        return True
    if word[0].isupper():
        return True
    return False


def score_fiction_names(text,
                          reward_per_exact = 3.0,
                          penalty_per_fuzzy = -1.0,
                          fuzzy_min = 70,
                          fuzzy_max = 92):

    words = re.findall(r"[A-Za-z']+", text)
    exact = 0
    fuzzy = 0
    for w in words:
        wl = w.lower()
        if wl in fiction_names:
            exact += 1
        else:
            dmin = word_match_score(wl, fiction_names)
            if fuzzy_min <= dmin <= fuzzy_max:
                fuzzy += 1
    score = exact * reward_per_exact + fuzzy * penalty_per_fuzzy
    return {"exact_hits": exact, "fuzzy_hits": fuzzy, "score": score}

_DIALOGUE_RE = re.compile(r'^\s*[“"]([^”"]+)[”"]\s*(?:,?\s*(said|replied|asked|whispered|muttered|shouted|cried)\b\s+[A-Z][a-z]+)?', re.I | re.M)

def score_dialogue_formatting(text,
                              reward_per_good_line = 2.0,
                              penalty_unmatched_quote = -2.0):

    lines = text.splitlines()
    good = 0
    unmatched = 0
    for ln in lines:
        if ln.strip() == "":
            continue
        
        quote_count = ln.count('"') + ln.count('“') + ln.count('”') + ln.count("�")
        # treat odd counts as unmatched
        if quote_count % 2 != 0:
            unmatched += 1
        # well-formed dialogue match (start with quote content and optionally 'said Name')
        if _DIALOGUE_RE.match(ln):
            good += 1
    score = good * reward_per_good_line + unmatched * penalty_unmatched_quote
    return {"good_lines": good, "unmatched_quotes": unmatched, "score": score}

def score_non_english_words(text,
    penalty = -1.0,
    min_len = 4,
    treat_names_as_ok = True):

    tokens = re.findall(r"[A-Za-z']+", text)
    non_english_count = 0
    examples = []

    for t in tokens:
        if len(t) <= min_len:
            continue
        if treat_names_as_ok and t[0].isupper():
            continue

        if t.lower() not in english_words:
            non_english_count += 1
            if len(examples) < 20:
                examples.append(t)

    score = non_english_count * penalty
    return {
        "non_english_count": non_english_count,
        "score": score,
        "examples": examples
    }






def get_final_score(s):
    score=score_fiction_names(s)['score']+score_dialogue_formatting(s)['score']+score_non_english_words(s)['score']
    return score
