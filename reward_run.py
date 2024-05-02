import torch
from tqdm.auto import tqdm
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
import torch
import collections
import random
tqdm.pandas()

from datasets import load_dataset, ClassLabel, load_metric, concatenate_datasets

from transformers import AutoModel, AutoTokenizer
from transformers import top_k_top_p_filtering
from torch import nn
from torch.nn import Identity
import torch.nn.functional as F
import torch

from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, AdamW, get_scheduler

from accelerate import Accelerator

from trainer.args import parse_args
from trainer.config import get_lora_config
from trainer.data import get_tokenizer
from trainer.data import load_data

import pandas as pd
from datasets import Dataset

def process_labels_function(examples):
    examples["label"] = [1 if (rating > 0.25) else 0 for rating in examples["rating"]]
    return examples

def process_text_function(examples):
    prompt_prefix = "Writing Prompt: "
    response_prefix = "Response: "
    examples["prompt"] = [prompt.replace('[WP] ', prompt_prefix) for prompt in examples["prompt"]]
    examples["response"] = [response_prefix + response for response in examples["response"]]
    return tokenizer(examples['prompt'], examples['response'], truncation=True)

def reward_train():
    # Parse arguments.
    args = parse_args()
    
    device = Accelerator().local_process_index
    # Tokenizer & dataset.
    tokenizer = get_tokenizer(args.tokenizer_name)
    
    # Load the dataset
    df = pd.read_csv('data/hotpotqa_rating.tsv', sep='\t').drop_duplicates(keep='last')
    dataset = Dataset.from_pandas(df)

    ## Balance our dataset (only select a small portion of the "not-best" labeled examples to match the number of best writing response examples)
    positive_reward_dataset = dataset.filter(lambda example: example['labels'] == 1)
    negative_reward_dataset = dataset.filter(lambda example: example['labels'] == 0).shuffle(seed=42).select(range(len(positive_reward_dataset)))
    reward_dataset = concatenate_datasets([positive_reward_dataset, negative_reward_dataset])
    # Reward model.
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_name,
    )
    reward_model = reward_model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.reward_model_name,
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    tokenized_datasets.set_format("torch")

    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=8)

    reward_model.to(device)

    optimizer = AdamW(reward_model.parameters(), lr=5e-5)

    num_epochs = 3

    for epoch in range(num_epochs):
        reward_model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            outputs = reward_model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    reward_model.save_pretrained("ckpts/reward_model")