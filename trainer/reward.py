import torch
from transformers import AutoModel
from transformers import AutoTokenizer


def reward_fn(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    prompt_text: list[str],
    response_text: list[str],
    device: str,
) -> list[torch.FloatTensor]:
    """Compute the reward for a given response to a prompt.

    Args:
        model (AutoModel): Huggingface model.
        tokenizer (AutoTokenizer): Huggingface tokenizer.
        prompt_text (list[str]): List of strings representing the prompt.
        response_text (list[str]): List of strings representing the response.
        device (str, optional): Device to run the model on. Defaults to 'cpu'.

    Returns:
        list[float]: A list of floats representing the reward.

    """
    with torch.no_grad():
        encoding = tokenizer(
            prompt_text,
            response_text,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt',
        )
        encoding = encoding.to(device)

        logits = model(**encoding).logits
        scores = logits.cpu().numpy().flatten().tolist()

        return scores


if __name__ == '__main__':
    prompt_text = [
        'The quick brown fox jumps over the lazy dog',
        'The quick brown fox jumps over the lazy dog',
    ]
    response_text = [
        'The quick brown fox jumps over the lazy dog',
        'The quick brown fox jumps over the lazy dog',
    ]
    device = 'cpu'

    # Using reward model.
    from transformers import AutoModelForSequenceClassification
    model_name = 'OpenAssistant/reward-model-deberta-v3-base'
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    scores = reward_fn(model, tokenizer, prompt_text, response_text, device)
    print(scores)