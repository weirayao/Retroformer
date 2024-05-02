from typing import Any

from peft import LoraConfig
from peft import TaskType
from .args import ScriptArgs
from trl import PPOConfig
from trl import RewardConfig

def get_reward_config(args: ScriptArgs) -> RewardConfig:
    """Get the reward model training config.
    
    Args:
        args: The script arguments.
    Returns:
        RewardConfig: The reward model training config.
    """
    
    config = RewardConfig(
        model_name=args.reward_model_name,
        steps=args.steps,
        learning_rate=args.learning_rate,
        log_with=args.log_with,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optimize_cuda_cache=True,
        early_stopping=args.early_stopping,
        seed=args.seed,
        tracker_project_name=args.project_name,
        tracker_kwargs={'run_name': args.run_name},
        max_length=args.max_length,
    )
    
    return config
    

def get_ppo_config(args: ScriptArgs) -> PPOConfig:
    """Get the PPO config.

    Args:
        args: The script arguments.

    Returns:
        PPOConfig: The PPO config.

    """
    config = PPOConfig(
        model_name=args.sft_model_name,
        steps=args.steps,
        learning_rate=args.learning_rate,
        log_with=args.log_with,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optimize_cuda_cache=True,
        early_stopping=args.early_stopping,
        target_kl=args.target_kl,
        ppo_epochs=args.ppo_epochs,
        seed=args.seed,
        init_kl_coef=args.init_kl_coef,
        adap_kl_ctrl=args.adap_kl_ctrl,
        tracker_project_name=args.project_name,
        tracker_kwargs={'run_name': args.run_name},
    )

    return config


def get_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    task_type: TaskType = TaskType.CAUSAL_LM,
    **kwargs: Any,
) -> LoraConfig:
    """Get the LoRA config.

    Args:
        r (int): The number of attention heads.
        lora_alpha (int): The LoRA alpha parameter.
        lora_dropout (float): The LoRA dropout parameter.
        task_type (TaskType): The task type.
        kwargs: Additional keyword arguments.

    Returns:
        LoraConfig: The LoRA config.

    """
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias='none',
        task_type=task_type,
        **kwargs,
    )

    return config