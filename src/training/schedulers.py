"""Learning rate scheduler factory.

Provides cosine, linear, and constant-with-warmup schedulers via the
HuggingFace transformers API.
"""

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_scheduler as hf_get_scheduler


def get_scheduler(
    name: str = "cosine",
    optimizer: Optimizer | None = None,
    num_warmup_steps: int = 0,
    num_training_steps: int = 1000,
) -> LambdaLR:
    """Factory function for learning rate schedulers.

    Supported names: 'cosine' (default), 'linear', 'constant'. Delegates
    to transformers.get_scheduler. Warmup is applied over num_warmup_steps steps.

    Args:
        name: Name of the scheduler type ('cosine', 'linear', 'constant').
        optimizer: PyTorch optimizer instance to wrap.
        num_warmup_steps: Number of linear warmup steps.
        num_training_steps: Total number of training optimization steps.

    Returns:
        Configured LambdaLR learning rate scheduler instance.
    """
    return hf_get_scheduler(
        name=name,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
