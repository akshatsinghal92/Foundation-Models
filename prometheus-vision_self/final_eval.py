import json
import re
import argparse
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from collections import Counter


SCORE_REGEX = re.compile(
    r"score\s+is\s+([1-5])",
    re.IGNORECASE
)


def extract_predicted_score(text):
    """
    Extracts score from:
    'So the overall score is X'
    """
    match = SCORE_REGEX.search(text)
    if match is None:
        raise ValueError(f"Could not extract score from text:\n{text}")
    return int(match.group(1))


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def main(pred_file, ref_file):
    preds = load_jsonl(pred_file)
    refs_data = load_jsonl(ref_file)
    
    # Handle the case where the reference file structure might be a list wrapped in a list
    # (based on original code refs = load_jsonl(ref_file)[0])
    if len(refs_data) == 1 and isinstance(refs_data[0], list):
        refs = refs_data[0]
    else:
        refs = refs_data

    # Map by question_id
    # Note: Using 'question_id' as the key. Ensure your data has this field.
    try:
        pred_map = {item['question_id']: item for item in preds}
        ref_map = {item['question_id']: item for item in refs}
    except KeyError as e:
        print(f"Error: Missing key {e} in data. Please check if 'question_id' exists.")
        return

    common_ids = set(pred_map.keys()) & set(ref_map.keys())
    print(f"Found {len(common_ids)} common question_ids.")

    if len(common_ids) == 0:
        print("No matching question_ids found between predictions and references.")
        return

    predicted_scores = []
    reference_scores = []

    for qid in common_ids:
        p = pred_map[qid]
        r = ref_map[qid]
        
        try:
            pred_score = extract_predicted_score(p["text"])
            ref_score = int(r["score"])
        except Exception as e:
            print(f"Error processing question_id {qid}: {e}")
            continue

        predicted_scores.append(pred_score)
        reference_scores.append(ref_score)

    predicted_scores = np.array(predicted_scores)
    reference_scores = np.array(reference_scores)

    # === Correlations ===
    pearson = pearsonr(predicted_scores, reference_scores)[0]
    spearman = spearmanr(predicted_scores, reference_scores)[0]
    kendall = kendalltau(predicted_scores, reference_scores)[0]

    # === Diagnostics ===
    print("\n===== SCORE DISTRIBUTION =====")
    print("Predicted:", Counter(predicted_scores))
    print("Reference:", Counter(reference_scores))

    mae = np.mean(np.abs(predicted_scores - reference_scores))

    print("\n===== CORRELATION RESULTS =====")
    print(f"Pearson   : {pearson:.4f}")
    print(f"Spearman  : {spearman:.4f}")
    print(f"Kendall   : {kendall:.4f}")
    print(f"MAE       : {mae:.4f}")

    print("\n===== DONE =====")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PROMETHEUS-style evaluator correlation computation"
    )
    parser.add_argument(
        "--pred",
        required=True,
        help="JSONL file with model outputs (contains 'text')"
    )
    parser.add_argument(
        "--ref",
        required=True,
        help="JSONL file with reference annotations (contains 'score')"
    )

    args = parser.parse_args()
    main(args.pred, args.ref)



