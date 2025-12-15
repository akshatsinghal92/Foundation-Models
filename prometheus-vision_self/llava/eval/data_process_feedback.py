import json
import re
import sys
# Removed pandas dependency

def parse_rubric(rubric_text):
    """
    Parses the rubric text to extract descriptions for scores 1 to 5.
    Rubric format is expected to be:
    Score 1: ...
    Score 2: ...
    ...
    """
    scores = {}
    pattern = re.compile(r"Score (\d): ([\s\S]*?)(?=\nScore \d:|$)", re.IGNORECASE)
    
    matches = pattern.findall(rubric_text)
    for score_num, description in matches:
        scores[int(score_num)] = description.strip()
    
    return {
        'orig_score1_description': scores.get(1, ""),
        'orig_score2_description': scores.get(2, ""),
        'orig_score3_description': scores.get(3, ""),
        'orig_score4_description': scores.get(4, ""),
        'orig_score5_description': scores.get(5, "")
    }

def format_text(row, rubric_parts):
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
        orig_instruction=row['instruction'],
        orig_response=row['response'],
        orig_reference_answer=row['reference_response'],
        orig_criteria="[Score Rubric]",
        orig_score1_description=rubric_parts['orig_score1_description'],
        orig_score2_description=rubric_parts['orig_score2_description'],
        orig_score3_description=rubric_parts['orig_score3_description'],
        orig_score4_description=rubric_parts['orig_score4_description'],
        orig_score5_description=rubric_parts['orig_score5_description']
    )

def main():
    input_file = "/home/839temp/prometheus-vision/new_test_data.json"
    output_file = "/home/839temp/prometheus-vision/new_ad_test_data.jsonl"
    
    print(f"Loading dataset from {input_file}...")
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        sys.exit(1)

    print(f"Loaded {len(data)} items. Converting to JSONL...")
    
    with open(output_file, 'w') as f:
        # Use simple progress indicator since tqdm might also be missing
        total = len(data)
        for i, row in enumerate(data):
            if i % 1000 == 0:
                print(f"Processing {i}/{total}...")
                
            rubric_parts = parse_rubric(row['rubric'])
            formatted_text = format_text(row, rubric_parts)
            
            # Using i as question_id since original idx was from pandas index
            item = {
                "question_id": i,
                "image": row['image'],
                "text": formatted_text
            }
            f.write(json.dumps(item) + "\n")

    print(f"Conversion complete. Saved to {output_file}")

if __name__ == "__main__":
    main()
