from datasets import load_dataset, Dataset

from transformers import AutoTokenizer, MambaForCausalLM, TrainingArguments, Trainer


MODEL_ID = "state-spaces/mamba-2.8b-hf"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = MambaForCausalLM.from_pretrained(MODEL_ID)

# Finetune on 1000 examples
dataset_path = 'finetune_dataset/combined_dataset'
loaded_dataset = Dataset.load_from_disk(dataset_path)

# Split the dataset into training and test sets
split_dataset = loaded_dataset.train_test_split(test_size=0.2)
# Access the split parts
train_dataset = split_dataset['train']
test_dataset = split_dataset['test']

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['query'], padding="max_length", truncation=True)

tokenized_dataset = split_dataset.map(tokenize_function, batched=True)

# Depending on your specific task, you might need to adjust the dataset format.
# For instance, creating labels for language modeling by shifting the inputs.
tokenized_dataset = tokenized_dataset.map(lambda examples: {'labels': examples['input_ids']}, batched=True)

# Freeze all parameters first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the parameters of the last few layers
for layer in model.backbone.layers[-2:]:  # Unfreezing the last 2 layers
    for param in layer.parameters():
        param.requires_grad = True

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",          # output directory for saving model checkpoints
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=64,    # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay regularization
    logging_dir="./logs",            # directory for storing logs
    logging_steps=10,
    evaluation_strategy="epoch",     # evaluate each `logging_steps`
    save_strategy='epoch',           # save the model every epoch
    load_best_model_at_end=True,     # load the best model when finished training
    metric_for_best_model='accuracy' # use accuracy to evaluate the best model
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

# Fine-tune the model
trainer.train()

save_directory = "finetunned_models/mamba"

# Save the model
model.save_pretrained(save_directory)

# Save the tokenizer
tokenizer.save_pretrained(save_directory)
