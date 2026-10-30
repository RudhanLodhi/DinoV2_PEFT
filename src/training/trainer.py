"""Custom HuggingFace Trainer for DINOv2 PEFT experiments.

Handles multi-label and multi-class tasks, routes to the correct loss,
and integrates with ExperimentLogger for per-epoch CSV logging.
"""

from typing import Any
import torch
import torch.nn as nn
from transformers import Trainer, TrainerCallback


class _EpochLoggingCallback(TrainerCallback):
    """Callback bridge dispatching epoch end events to DINOv2PEFTTrainer."""

    def __init__(self, trainer: "DINOv2PEFTTrainer") -> None:
        """Initializes callback bridge with parent trainer reference.

        Args:
            trainer: Parent DINOv2PEFTTrainer instance.
        """
        self.trainer: DINOv2PEFTTrainer = trainer

    def on_epoch_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        """Dispatches epoch end event to trainer method.

        Args:
            args: Training arguments.
            state: Trainer state.
            control: Trainer control.
            **kwargs: Extra event arguments.
        """
        self.trainer.on_epoch_end(args, state, control, **kwargs)


class DINOv2PEFTTrainer(Trainer):
    """Custom HuggingFace Trainer with task-specific loss routing and CSV logging."""

    def __init__(
        self,
        task: str,
        loss_fn: nn.Module,
        experiment_logger: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Extends Trainer with task-aware loss routing and experiment logging.

        task must be 'multilabel' (CheXpert) or 'multiclass' (Brain Tumor).
        loss_fn is the appropriate loss function. experiment_logger is an optional
        ExperimentLogger instance for CSV logging.

        Args:
            task: Task identifier ('multilabel' or 'multiclass').
            loss_fn: Appropriate loss module for the task.
            experiment_logger: Optional ExperimentLogger instance.
            **kwargs: Additional keyword arguments forwarded to HuggingFace Trainer.
        """
        super().__init__(**kwargs)
        self.task: str = task
        self.loss_fn: nn.Module = loss_fn
        self.experiment_logger: Any | None = experiment_logger

        if self.experiment_logger is not None:
            self.add_callback(_EpochLoggingCallback(self))

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Routes forward pass through backbone and classification head, computing task loss.

        For multilabel: extracts CLS token (last_hidden_state[:,0,:]), passes through head,
        applies BCEWithLogitsLoss. For multiclass: same flow with CrossEntropyLoss.

        Args:
            model: PyTorch model wrapping backbone and classification head.
            inputs: Batch dictionary containing input tensors and labels.
            return_outputs: Whether to return prediction logits alongside the loss.

        Returns:
            Computed scalar loss tensor, or tuple of (loss, logits) if return_outputs is True.
        """
        pixel_values: torch.Tensor = inputs["pixel_values"]
        if self.task == "multilabel":
            targets: torch.Tensor = inputs.get("labels", inputs.get("label"))
        else:
            targets = inputs.get("label", inputs.get("labels"))

        if hasattr(model, "backbone") and hasattr(model, "head"):
            outputs = model.backbone(pixel_values)
            cls_token: torch.Tensor = outputs.last_hidden_state[:, 0, :]
            logits: torch.Tensor = model.head(cls_token)
        else:
            outputs = model(pixel_values)
            if isinstance(outputs, torch.Tensor):
                logits = outputs
            elif hasattr(outputs, "logits"):
                logits = outputs.logits
            elif hasattr(outputs, "last_hidden_state"):
                logits = outputs.last_hidden_state[:, 0, :]
            else:
                logits = outputs

        loss: torch.Tensor = self.loss_fn(logits, targets)
        return (loss, logits) if return_outputs else loss

    def on_epoch_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> None:
        """Called at end of each training epoch.

        If experiment_logger is set, logs the current epoch, learning rate,
        and training loss to the experiment CSV.

        Args:
            args: Training arguments.
            state: Trainer state.
            control: Trainer control.
            **kwargs: Extra event arguments.
        """
        if self.experiment_logger is not None and state is not None:
            current_lr: float = 0.0
            if self.optimizer is not None and len(self.optimizer.param_groups) > 0:
                current_lr = float(self.optimizer.param_groups[0].get("lr", 0.0))

            training_loss: float = 0.0
            if state.log_history:
                for entry in reversed(state.log_history):
                    if "loss" in entry:
                        training_loss = float(entry["loss"])
                        break

            epoch_val: float = float(state.epoch) if state.epoch is not None else 0.0
            self.experiment_logger.log(
                notes=f"Epoch {epoch_val:.1f}: train_loss={training_loss:.4f}, lr={current_lr:.6e}"
            )
