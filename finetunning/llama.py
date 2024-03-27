from datasets import Dataset

from transformers import TrainingArguments, LlamaTokenizer, LlamaForCausalLM
from trl import SFTTrainer
from peft import LoraConfig

MODEL_ID = "openlm-research/open_llama_3b_v2"

tokenizer = LlamaTokenizer.from_pretrained(MODEL_ID)
tokenizer.add_eos_token = True
tokenizer.pad_token = tokenizer.eos_token

model = LlamaForCausalLM.from_pretrained(MODEL_ID)

# Finetune on 1000 examples
dataset_path = 'finetune_dataset/combined_dataset'
loaded_dataset = Dataset.load_from_disk(dataset_path)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['query'], padding="max_length", truncation=True, max_length=500)

tokenized_dataset = loaded_dataset.map(tokenize_function, batched=True)

tokenized_dataset = tokenized_dataset.map(lambda examples: {'labels': examples['input_ids']}, batched=True)
# Split the dataset into training and test sets
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./finetunned_models/llama/results",          # output directory for saving model checkpoints
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=8,    # batch size for evaluation
    warmup_steps=100,                # number of warmup steps for learning rate scheduler
    logging_dir="./logs",            # directory for storing logs
    logging_steps=10,
    learning_rate=2e-3,
    evaluation_strategy="epoch",     # evaluate each `logging_steps`
    save_strategy='epoch',           # save the model every epoch
    load_best_model_at_end=True,     # load the best model when finished training
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    task_type="CAUSAL_LM",
    bias="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=lora_config,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    dataset_text_field="query",
)

# Fine-tune the model
trainer.train()

save_directory = "finetunned_models/llama"

# Save the model
model.save_pretrained(save_directory)

# Save the tokenizer
tokenizer.save_pretrained(save_directory)
