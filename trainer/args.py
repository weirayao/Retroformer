from dataclasses import dataclass
from dataclasses import field
from datetime import datetime as dt

from transformers import HfArgumentParser


@dataclass
class ScriptArgs:
    """The name of the Causal LM wwe wish to fine-tune with PPO."""

    sft_model_name: str | None = field(
        default='',
        metadata={
            'help': 'The name of the SFT model we wish to fine-tune with PPO.',
        },
    )

    tokenizer_name: str | None = field(
        default='',
        metadata={
            'help': 'The name of the tokenizer to use.',
        },
    )

    reward_model_name: str | None = field(
        default='',
        metadata={
            'help': 'The name of the reward model to use.',
        },
    )

    ppo_model_name: str | None = field(
        default='',
        metadata={
            'help': 'The name of the PPO model to use for evaluation only.',
        },
    )

    dataset_path: str | None = field(
        default='res/data/',
        metadata={
            'help': 'The path to the dataset.',
        },
    )

    run_name: str | None = field(
        default=f'rlhf-{dt.now():%d-%m-%Y-%H_%M_%S}',
        metadata={
            'help': 'The name of the experiment.',
        },
    )

    eval_save_path: str | None = field(
        default='res/eval/',
        metadata={
            'help': 'The path to save the evaluation results.',
        },
    )

    eval_name: str | None = field(
        default='rlhf_eval',
        metadata={
            'help': 'The name of the evaluation file without extension.',
        },
    )

    output_dir: str | None = field(
        default='experiments',
        metadata={
            'help': 'The output directory.',
        },
    )

    project_name: str | None = field(
        default='RLHF-TRL',
        metadata={
            'help': 'The name of the tracker project.',
        },
    )

    log_with: str | None = field(
        default='wandb',
        metadata={
            'help': 'The logger to use.',
        },
    )

    learning_rate: float | None = field(
        default=1e-5,
        metadata={
            'help': 'The learning rate to use.',
        },
    )

    output_max_length: int | None = field(
        default=256,
        metadata={
            'help': 'The maximum length of the output sequence for generation.',
        },
    )

    output_min_length: int | None = field(
        default=32,
        metadata={
            'help': 'The minimum length of the output sequence for generation.',
        },
    )

    mini_batch_size: int | None = field(
        default=1,
        metadata={
            'help': 'The mini-batch size to use for PPO.',
        },
    )

    batch_size: int | None = field(
        default=8,
        metadata={
            'help': 'The batch size to use for PPO.',
        },
    )

    ppo_epochs: int | None = field(
        default=50,
        metadata={
            'help': 'The number of PPO epochs.',
        },
    )

    gradient_accumulation_steps: int | None = field(
        default=2,
        metadata={
            'help': 'The number of gradient accumulation steps.',
        },
    )

    adafactor: bool | None = field(
        default=False,
        metadata={
            'help': 'Whether to use AdaFactor instead of AdamW.',
        },
    )

    early_stopping: bool | None = field(
        default=False,
        metadata={
            'help': 'Whether to use early stopping.',
        },
    )

    target_kl: float | None = field(
        default=0.1,
        metadata={
            'help': 'The target KL divergence for early stopping.',
        },
    )

    reward_baseline: float | None = field(
        default=0.0,
        metadata={
            'help': 'The reward baseline is a value subtracted from the reward.',
        },
    )

    batched_gen: bool | None = field(
        default=False,
        metadata={
            'help': 'Whether to use batched generation.',
        },
    )

    save_freq: int | None = field(
        default=10,
        metadata={
            'help': 'The frequency with which to save the model.',
        },
    )

    seed: int | None = field(
        default=42,
        metadata={
            'help': 'The seed.',
        },
    )

    steps: int | None = field(
        default=100_000,
        metadata={
            'help': 'The number of training steps.',
        },
    )

    lr_gamma: float | None = field(
        default=0.9,
        metadata={
            'help': 'The learning rate scheduler gamma.',
        },
    )

    init_kl_coef: float | None = field(
        default=0.2,
        metadata={
            'help': 'The initial KL coefficient (used for adaptive & linear control.',
        },
    )

    adap_kl_ctrl: bool | None = field(
        default=True,
        metadata={
            'help': 'Whether to adaptively control the KL coefficient, linear otherwise.',
        },
    )
    
    max_length: int | None = field(
        default=16384,
        metadata={
            'help': 'The maximum length of the input sequence.',
        },
    )


def parse_args() -> ScriptArgs:
    """Parse the command line arguments.

    Returns:
        ScriptArgs: The parsed command line arguments.

    """
    parser = HfArgumentParser(ScriptArgs)
    args = parser.parse_args_into_dataclasses()[0]
    return args