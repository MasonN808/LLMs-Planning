from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig

from src.modeling_mamba import MambaForCausalLM
from transformers import AutoTokenizer


MODEL_ID = 'Q-bert/Mamba-130M'

model = MambaForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

text = "Hi"

# input_ids = tokenizer.encode(text, return_tensors="pt")

# output = model.generate(input_ids, max_length=20, num_beams=5, no_repeat_ngram_size=2)

# generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# print(generated_text)
# exit()

# MODEL_ID = "state-spaces/mamba-130m-hf"

# tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# model = MambaForCausalLM.from_pretrained(MODEL_ID)
# exit()

# input_ids = tokenizer("Hey how are you doing?", return_tensors="pt")["input_ids"]

# out = model.generate(input_ids, max_new_tokens=10)
# print(tokenizer.batch_decode(out))

dataset = load_dataset("chiayewken/blocksworld", split="train")
print(dataset)
exit()
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=2e-3
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
    train_dataset=dataset,
    dataset_text_field="quote",
)
trainer.train()

save_directory = "models/mamba"

# Save the model
model.save_pretrained(save_directory)

# Save the tokenizer
tokenizer.save_pretrained(save_directory)
