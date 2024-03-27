from datasets import load_dataset, Dataset

from transformers import AutoTokenizer, MambaForCausalLM, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig

MODEL_ID = "state-spaces/mamba-2.8b-hf"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = MambaForCausalLM.from_pretrained(MODEL_ID)

# Finetune on 1000 examples
dataset_path = 'finetune_dataset/combined_dataset'
loaded_dataset = Dataset.load_from_disk(dataset_path)

# Tokenize the datasete
def tokenize_function(examples):
    # return tokenizer(examples['query'], padding="max_length", truncation=True, max_length=500)
    return tokenizer(examples['query'], truncation=True)

tokenized_dataset = loaded_dataset.map(tokenize_function, batched=True)

tokenized_dataset = tokenized_dataset.map(lambda examples: {'labels': examples['input_ids']}, batched=True)
# Split the dataset into training and test sets
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./finetunned_models/mamba/results",          # output directory for saving model checkpoints
    num_train_epochs=2,              # number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=64,    # batch size for evaluation
    warmup_steps=50,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay regularization
    logging_dir="./logs",            # directory for storing logs
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

save_directory = "finetunned_models/mamba"

# Save the model
model.save_pretrained(save_directory)

# Save the tokenizer
tokenizer.save_pretrained(save_directory)
