from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig, HfArgumentParser, TrainingArguments

from trl import SFTTrainer

from fastchat.model.llama_condense_monkey_patch import replace_llama_with_condense

import pandas as pd
from datasets import Dataset

tqdm.pandas()


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    model_name: Optional[str] = field(default="lmsys/longchat-7b-16k", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco", metadata={"help": "the dataset name"}
    )
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=2, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=16384, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=8, metadata={"help": "the number of gradient accumulation steps"}
    )
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=True, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=True, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="output/sft/", metadata={"help": "the output directory"})
    peft_lora_r: Optional[int] = field(default=2, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the number of logging steps"})
    use_auth_token: Optional[bool] = field(default=True, metadata={"help": "Use HF auth token to access the model"})
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Step 1: Load the model
auto_config = AutoConfig.from_pretrained(script_args.model_name)
replace_llama_with_condense(auto_config.rope_condense_ratio)
    
model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    load_in_4bit=True,
    low_cpu_mem_usage=True,
    device_map="balanced")

tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
# GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
# only for this model.

if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.eos_token

# Step 2: Load the dataset
df = pd.read_csv('data/hotpotqa_rating.tsv', sep='\t').drop_duplicates(keep='last')
train_dataset = Dataset.from_pandas(df)

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['query'])):
        text = f"{example['query'][i]} {example['response'][i]}"
        output_texts.append(text)
    return output_texts

# Step 3: Define the training arguments
training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    save_steps=25,
    report_to="wandb",
    num_train_epochs=script_args.num_train_epochs,
    max_steps=script_args.max_steps,
)

# Step 4: Define the LoraConfig
if script_args.use_peft:
    peft_config = LoraConfig(
        r=script_args.peft_lora_r,
        lora_alpha=script_args.peft_lora_alpha,
        bias="none",
        task_type="CAUSAL_LM",
    )
else:
    peft_config = None

# Step 5: Define the Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    formatting_func=formatting_prompts_func,
    peft_config=peft_config,
    max_seq_length=script_args.seq_length,
)

trainer.train()
