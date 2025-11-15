import json
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset, DatasetDict
import os

os.makedirs("../weights", exist_ok=True)

# ---- Load preprocessed datasets ----
with open("../data/games.json") as f:
    games = json.load(f)

with open("../data/puzzles.json") as f:
    puzzles = json.load(f)

# Combine both datasets
combined = games + puzzles

# Convert to Hugging Face Dataset
ds = Dataset.from_list(combined)
ds = ds.shuffle(seed=42)

# Tokenizer + Model
model_name = "gpt2"  # Small transformer to start
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({"pad_token": "<PAD>"})

model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

# Preprocess function
def encode(example):
    if example["type"] == "game":
        text = " ".join(example["moves"])
    else:
        # puzzle: represent as FEN + solution moves
        text = example["fen"] + " " + " ".join(example["solution"])
    return tokenizer(text, truncation=True, padding="max_length", max_length=128)

ds = ds.map(encode, batched=False)

# Split into train / validation
split = ds.train_test_split(test_size=0.1)
dataset_dict = DatasetDict({
    "train": split["train"],
    "validation": split["test"]
})

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training args
training_args = TrainingArguments(
    output_dir="../weights",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=200,
    learning_rate=5e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Start training
trainer.train()

# Save final model
trainer.save_model("../weights/ash")
tokenizer.save_pretrained("../weights/ash")
print("Model saved to weights/ash")
