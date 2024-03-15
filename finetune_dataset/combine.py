import json
from datasets import Dataset, Features, Value
import os

"""
This file combines all of the prompts into a single dataset for finetunning
"""

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

# Initialize separate lists for each column
queries = []
ground_truth_plans = []
print(json_file_paths)
# Loop through each file path in the list
for file_path in json_file_paths:
    with open(file_path, 'r') as file:
        data = json.load(file)
        for instance in data['instances']:
            # Extract the query
            queries.append(instance['query'])
            
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
dataset_path = 'finetune_dataset/combined_dataset'
dataset.save_to_disk(dataset_path)
