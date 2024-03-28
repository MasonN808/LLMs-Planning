from datasets import Dataset

# Load the dataset for the current task
loaded_dataset = Dataset.load_from_disk('finetune_dataset/separate_datasets/t0_dataset')
print(loaded_dataset[0])