from datasets import load_dataset, Dataset

from transformers import AutoTokenizer, MambaForCausalLM, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig

MODEL_ID = "state-spaces/mamba-2.8b-hf"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

task_names = [
    'task_1_plan_generation',
    'task_2_plan_optimality',
    'task_3_plan_verification',
    'task_4_plan_reuse',
    'task_5_plan_generalization',
    'task_6_replanning',
    'task_7_plan_execution',
    'task_8_1_goal_shuffling',
    'task_8_2_full_to_partial',
    'task_8_3_partial_to_full'
]
# Corresponding dataset paths
dataset_paths = [
    'finetune_dataset/separate_datasets/t0_dataset',
    'finetune_dataset/separate_datasets/t1_dataset',
    'finetune_dataset/separate_datasets/t2_dataset',
    'finetune_dataset/separate_datasets/t3_dataset',
    'finetune_dataset/separate_datasets/t4_dataset',
    'finetune_dataset/separate_datasets/t5_dataset',
    'finetune_dataset/separate_datasets/t6_dataset',
    'finetune_dataset/separate_datasets/t7_dataset',
    'finetune_dataset/separate_datasets/t8_dataset',
    'finetune_dataset/separate_datasets/t9_dataset'
]

def tokenize_function(examples):
    # return tokenizer(examples['query'], padding="max_length", truncation=True, max_length=500)
    return tokenizer(examples['query'], truncation=True)

for task_name, dataset_path in zip(task_names, dataset_paths):
    # Load the dataset for the current task
    loaded_dataset = Dataset.load_from_disk(dataset_path)

    # Tokenize the datasete
    tokenized_dataset = loaded_dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.map(lambda examples: {'labels': examples['input_ids']}, batched=True)

    # Split the dataset into training and test sets
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)

    # Load the base model afresh for each task to avoid cumulative training across tasks
    model = MambaForCausalLM.from_pretrained(MODEL_ID)
    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=f"./finetuned_models/mamba/results/{task_name}",          # output directory for saving model checkpoints
        num_train_epochs=3,              # number of training epochs
        per_device_train_batch_size=8,   # batch size for training
        per_device_eval_batch_size=64,    # batch size for evaluation
        warmup_steps=10,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay regularization
        logging_dir=f"./logs/{task_name}",            # directory for storing logs
        logging_steps=10,
        learning_rate=2e-3,
        evaluation_strategy="epoch",     # evaluate each `logging_steps`
        save_strategy='epoch',           # save the model every epoch
        load_best_model_at_end=True,     # load the best model when finished training
    )

    lora_config =  LoraConfig(
            r=8,
            target_modules=["x_proj", "embeddings", "in_proj", "out_proj"],
            task_type="CAUSAL_LM",
            bias="none"
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        peft_config=lora_config,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        dataset_text_field="query"
    )

    # Fine-tune the model
    trainer.train()

    # Save the model and tokenizer for the current task
    save_directory = f"./finetuned_models/mamba/{task_name}"
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
