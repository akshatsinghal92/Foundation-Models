import json
import os

def process_data(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    processed_data = []
    
    # Template parts
    task_description = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing an evaluation criterion is given.
1. Write a detailed feedback that assesses the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)
4. Please do not generate any other opening, closing, and explanations."""

    for i, item in enumerate(data):
        # Extract fields
        image_file = item.get('image', '')
        instruction = item.get('instruction', '')
        response = item.get('response', '')
        reference_answer = item.get('reference_response', '')
        rubric = item.get('rubric', '')
        feedback = item.get('feedback', '')
        score = item.get('score', '')
        
        # Construct human prompt
        # Note: The template in the prompt had specific placeholders. 
        # We need to reconstruct the "Score Rubrics" section.
        # Since the input 'rubric' field seems to contain the "Score 1: ... Score 2: ..." text directly,
        # we will use it.
        # However, the example output has a question/criteria text BEFORE the scores.
        # The input JSON does NOT seem to have this criteria text separate or at all.
        # We will append the rubric text as is.
        
        human_value = f"""<image>
{task_description}

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
{rubric}

###Feedback:"""

        # Construct GPT response
        gpt_value = f"{feedback}\n[RESULT] {score}"

        # Build entry
        entry = {
            "id": i,
            "image": f"../images/{image_file}",
            "conversations": [
                {
                    "from": "human",
                    "value": human_value
                },
                {
                    "from": "gpt",
                    "value": gpt_value
                }
            ]
        }
        processed_data.append(entry)

    with open(output_file, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    print(f"Processed {len(processed_data)} items. Output saved to {output_file}")

if __name__ == "__main__":
    input_path = "/home/839temp/prometheus-vision/formatted_responses_feedback.json"
    output_path = "/home/839temp/prometheus-vision/llava_judge_data_output.json"
    process_data(input_path, output_path)
