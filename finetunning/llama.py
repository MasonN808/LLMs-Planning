# Motivated by https://www.run.ai/guides/generative-ai/llama-2-fine-tuning

import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline, logging
# Assuming LoraConfig and SFTTrainer are available in your environment
from peft import LoraConfig
from trl import SFTTrainer
import gc

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token="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")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", token="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")
model.config.use_cache = True

# PyTorch memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
gc.collect()
torch.cuda.empty_cache()

def display_cuda_memory():
    print("\n--------------------------------------------------\n")
    print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(0)/1024/1024/1024))
    print("\n--------------------------------------------------\n")

# Define model, dataset, and new model name
base_model = "meta-llama/Llama-2-7b-hf"
blocksworld_dataset = "chiayewken/blocksworld"
new_model = "llama-2-7b-finetunned"

# Load dataset
dataset = load_dataset(blocksworld_dataset, split="train")
print(dataset[:5])
exit()
# 4-bit Quantization Configuration
compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=False)

# Load model with 4-bit precision
model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=quant_config, device_map={"": 0})
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Set PEFT Parameters
peft_params = LoraConfig(lora_alpha=16, lora_dropout=0.1, r=64, bias="none", task_type="CAUSAL_LM")

# Define training parameters
training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="adamw",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    report_to="tensorboard"
)

# Initialize the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False
)

# Train the model
trainer.train()

# Save the model and tokenizer
trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)

# Test the model
logging.set_verbosity(logging.CRITICAL)
prompt = "Who is Leonardo Da Vinci?"
pipe = pipeline(task="text-generation", model=new_model, tokenizer=new_model, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])
