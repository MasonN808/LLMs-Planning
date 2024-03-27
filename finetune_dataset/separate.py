import json
from datasets import Dataset, Features, Value
import os

"""
This file separates all of the prompts into a single dataset for finetunning
"""

def extract_up_to(s):
    keyword_plan_end = '[PLAN END]'
    keyword_action_end = '[ACTION SEQUENCE END]'
    
    # Ensure 'query' is a string
    if not isinstance(s, str):
        # Handle non-string 'query' appropriately
        raise TypeError(f"Expected a string for 'query', but got: {type(s)}")

    # Find the index of both keywords
    index_plan_end = s.rfind(keyword_plan_end)
    index_action_end = s.rfind(keyword_action_end)
    
    # Determine which keyword to use based on their presence and position
    if index_plan_end != -1 and (index_action_end == -1 or index_plan_end < index_action_end):
        end_index = index_plan_end + len(keyword_plan_end)
    elif index_action_end != -1:
        end_index = index_action_end + len(keyword_action_end)
    else:
        # Handle case where neither keyword is found
        # You can choose to return the original string, raise an error, or handle it another way
        raise ValueError("Neither '[PLAN END]' nor '[ACTION SEQUENCE END]' found in the string.")

    # Extract and return the substring up to and including the relevant keyword
    s = s[:end_index]
    return s

domain = "blocksworld_3"
# Get all json files in domain
directory_path = f'finetune_dataset/prompts/{domain}'

# Initialize an empty list to hold the paths to JSON files
json_file_paths = []

# Loop through each file in the directory
for file in os.listdir(directory_path):
    # Check if the file is a JSON file
    if file.endswith(".json"):
        # Construct the full file path and add it to the list
        json_file_paths.append(os.path.join(directory_path, file))

# Loop through each file path in the list
for idx, file_path in enumerate(json_file_paths):
    # Initialize separate lists for each column
    queries = []
    ground_truth_plans = []
    with open(file_path, 'r') as file:
        data = json.load(file)
        for instance in data['instances']:
            # Extract the query
            queries.append(extract_up_to(instance['query']))
            
            # Check if ground_truth_plan is a list and join into a single string if necessary
            if isinstance(instance['ground_truth_plan'], list):
                ground_truth_plans.append('\n'.join(instance['ground_truth_plan']))
            else:
                ground_truth_plans.append(instance['ground_truth_plan'])

    data_dict = {
        'query': queries,
        'ground_truth_plan': ground_truth_plans
    }

    features = Features({
        'query': Value('string'),
        'ground_truth_plan': Value('string')
    })

    dataset = Dataset.from_dict(data_dict, features=features)

    # Save the dataset locally as a JSON file (you can choose other formats like csv)
    dataset_path = f'finetune_dataset/separate_datasets/t{idx}_dataset'
    dataset.save_to_disk(dataset_path)
