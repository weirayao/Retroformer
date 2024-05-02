import os
import time

import torch
from trainer.args import parse_args
from trainer.data import collator
from trainer.data import get_tokenizer
from trainer.data import load_data
from trainer.reward import reward_fn
from trainer.ppo_trainer import build_trainer
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from trl.core import LengthSampler
    
def ppo_train() -> None:
    """Train the model."""

    # Parse arguments.
    args = parse_args()

    # Tokenizer & dataset.
    tokenizer = get_tokenizer(args.tokenizer_name)
    dataset = load_data(args.dataset_path, tokenizer, split='all')

    # PPO Trainer.
    _, ppo_trainer = build_trainer(
        args=args,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
    )

    gen_kwargs = {
        'top_k': 0.0,
        'top_p': 0.9,
        'do_sample': True,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
    }
    output_length_sampler = LengthSampler(
        args.output_min_length,
        args.output_max_length,
    )

    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = (
            'cuda' if torch.cuda.is_available()
            else 'mps'
            if torch.backends.mps.is_available() and torch.backends.mps.is_built()
            else 'cpu',
        )

    # Reward model.
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_name,
    )
    reward_model = reward_model.to(device)
    reward_tokenizer = AutoTokenizer.from_pretrained(
        args.reward_model_name,
    )

    ppo_trainer.accelerator.print(f'Using device: {device}')
    start_time = time.time()
    # Training loop.
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader), desc='Training PPO'):
        # if epoch >= config.total_ppo_epochs:
        #     break

        prompt_tensors = batch['input_ids']
        response_tensors = ppo_trainer.generate(
            prompt_tensors,
            return_prompt=False,
            length_sampler=output_length_sampler,
            **gen_kwargs,
        )
        batch['response'] = tokenizer.batch_decode(
            response_tensors,
            skip_special_tokens=True,
        )

        # Compute reward score.
        scores = reward_fn(
            model=reward_model,
            tokenizer=reward_tokenizer,
            prompt_text=batch['prompt'],
            response_text=batch['response'],
            device=device,
        )
        rewards = [
            torch.tensor(score - args.reward_baseline)
            for score in scores
        ]

        # Run the PPO step.
        stats = ppo_trainer.step(prompt_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

        # Save the model.
        if args.save_freq and epoch and epoch % args.save_freq == 0:
            ppo_trainer.save_pretrained(
                os.path.join(
                    args.output_dir,
                    args.project_name,
                    args.run_name,
                    f'checkpoint-{epoch}',
                ),
            )

    elapsed_time = time.time() - start_time
    mins, secs = divmod(elapsed_time, 60)
    hours, mins = divmod(mins, 60)
    ppo_trainer.accelerator.print(f'Training took {hours:.0f}h {mins:.0f}m {secs:.0f}s.')

    ppo_trainer.accelerator.print('\nSaving model!')
    ppo_trainer.save_pretrained(
        os.path.join(
            args.output_dir,
            args.project_name,
            args.run_name,
            'model',
        ),
    )


if __name__ == '__main__':
    # To train the reward model, run the following command:
    
    # To train the PPO model, run the following command:
    ppo_train()