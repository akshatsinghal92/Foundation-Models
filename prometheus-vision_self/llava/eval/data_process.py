from datasets import load_dataset
import pandas as pd
import json
from tqdm import tqdm

def format_text(row):
    template = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, image and a score rubric representing an evaluation criterion is given.
1. Write a detailed feedback that assesses the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{orig_instruction}

###Response to evaluate:
{orig_response}

###Reference Answer (Score 5):
{orig_reference_answer}

###Score Rubrics:
[{orig_criteria}]
Score 1: {orig_score1_description}
Score 2: {orig_score2_description}
Score 3: {orig_score3_description}
Score 4: {orig_score4_description}
Score 5: {orig_score5_description}

###Feedback:
"""
    return template.format(
        orig_instruction=row['orig_instruction'],
        orig_response=row['orig_response'],
        orig_reference_answer=row['orig_reference_answer'],
        orig_criteria=row['orig_criteria'],
        orig_score1_description=row['orig_score1_description'],
        orig_score2_description=row['orig_score2_description'],
        orig_score3_description=row['orig_score3_description'],
        orig_score4_description=row['orig_score4_description'],
        orig_score5_description=row['orig_score5_description']
    )

print("Loading dataset...")
ds = load_dataset("prometheus-eval/Perception-Collection")
df = pd.DataFrame(ds["train"])

df=df[df['image'].str.contains("mmmu")]

df = df.sample(n=150)

print("Converting to JSONL...")
output_file = "perception_eval_small.jsonl"
with open(output_file, 'w') as f:
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        item = {
            "question_id": idx,
            "image": row['image'],
            "text": format_text(row)
        }
        f.write(json.dumps(item) + "\n")

print(f"Conversion complete. Saved to {output_file}")
print(df.head())
