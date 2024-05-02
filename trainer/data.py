import os
from typing import Any

import jsonlines
import tqdm
from datasets import Dataset
from datasets import Split
from transformers import AutoTokenizer


def load_data(
    path: str,
    tokenizer: AutoTokenizer,
    split: str = 'train',
    return_answers: bool = False,
    max_size: int | None = None,
    max_token: int = 1024,
) -> Dataset:
    """Load the OpenAssistant dataset.

    Args:
        path: Path to the dataset.
        split: Split to load.
        max_size: Maximum number of examples to load.
            Defaults to None.

    Returns:
        Dataset: The dataset.

    """
    assert split in ['train', 'test', 'all'], 'split must be either train, test or all.'

    path = os.path.join(path, f'{split}.jsonl')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} does not exist.')

    with jsonlines.open(path) as reader:
        data = [obj for obj in reader]

    prompts, input_ids = [], []
    if return_answers:
        oa_ans, cgpt_ans = [], []

    qa_prompt: str = '<|prompter|>{}<|endoftext|><|assistant|>'

    for obj in tqdm.tqdm(
        data,
        total=len(data) if max_size is None else max_size,
        desc='Loading data',
    ):
        prompt = qa_prompt.format(obj['prompt'])
        prompts.append(prompt)

        tokenized_prompt = tokenizer(prompt, truncation=True)
        input_ids.append(tokenized_prompt['input_ids'])

        if return_answers:
            oa_ans.append(obj['openassistant-answer'])
            cgpt_ans.append(obj['chatgpt-answer'])

        if max_size is not None and len(prompts) >= max_size:
            break

    if return_answers:
        mapping = {
            'query': prompts,
            'input_ids': input_ids,
            'openassistant-answer': oa_ans,
            'chatgpt-answer': cgpt_ans,
        }
    else:
        mapping = {
            'query': prompts,
            'input_ids': input_ids,
        }

    split = 'train' if split == 'all' else split
    ds = Dataset.from_dict(
        mapping,
        split=Split.TRAIN if split == 'train' else Split.TEST,
    )

    ds = ds.filter(lambda x: len(x['input_ids']) <= max_token, batched=False)
    ds.set_format(type='torch')  # , columns=['input_ids'])

    return ds


def get_tokenizer(
    tokenizer_name: str,
    pad_token_as_eos: bool = True,
    padding_side: str | None = None,
) -> AutoTokenizer:
    """Get the tokenizer.

    Args:
        tokenizer_name: Name of the tokenizer.
        pad_token_as_eos: Whether to use the pad token as the eos token.
            Defaults to True.

    Returns:
        AutoTokenizer: The tokenizer.

    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if pad_token_as_eos and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if padding_side is not None:
        assert padding_side in ['left', 'right'], 'padding_side must be either left or right.'
        tokenizer.padding_side = padding_side

    return tokenizer


# def collator(
#     tokenizer: AutoTokenizer,
#     max_token: int = 1024,
# ) -> Callable[..., Any]:
#     """Collator function for the dataset.
#
#     Args:
#         tokenizer: The tokenizer.
#         max_token: Maximum number of tokens in the input.
#             Defaults to 1024.
#
#     Returns:
#         callable: The collator function.
#
#     """
#     def collate_fn(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
#         input_ids = [obj['input_ids'] for obj in batch]
#         input_ids = tokenizer.pad(
#             input_ids,
#             padding=True,
#             max_length=max_token,
#             return_tensors='pt',
#         )
#
#         return input_ids
#
#     return collate_fn


def collator(data: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Collator function for the dataset.

    Args:
        data (list[dict[str, Any]]): List of dictionaries.

    """
    return {key: [d[key] for d in data] for key in data[0]}