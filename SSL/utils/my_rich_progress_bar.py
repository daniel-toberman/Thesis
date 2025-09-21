import sys
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import (
    RichProgressBarTheme
)
from torch import Tensor


class MyRichProgressBar(RichProgressBar):
    """A progress bar that prints metrics at the end of each epoch"""

    def __init__(self, *args, **kwargs):
        # Provide a default theme if none was given
        if "theme" not in kwargs or kwargs["theme"] is None:
            kwargs["theme"] = RichProgressBarTheme(
                description="bold cyan",
                progress_bar="green",
                progress_bar_finished="green",
                progress_bar_pulse="#6206E0",
                batch_progress="green",
                time="grey82",
                processing_speed="grey82",
                metrics="grey82",
            )
        super().__init__(*args, **kwargs)

    def _init_progress(self, trainer):
        """Override to safely handle console initialization"""
        try:
            super()._init_progress(trainer)
        except (IndexError, AttributeError):
            # Initialize progress manually if needed
            from rich.progress import Progress, TaskID
            if not hasattr(self, 'progress') or self.progress is None:
                self.progress = Progress()
                self.main_progress_task_id = self.progress.add_task("Training", total=100)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        """Override to safely handle progress assertion"""
        if hasattr(self, 'progress') and self.progress is not None:
            super().on_validation_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Override to safely handle progress assertion"""
        if hasattr(self, 'progress') and self.progress is not None:
            super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Override to safely handle progress assertion"""
        if hasattr(self, 'progress') and self.progress is not None:
            super().on_train_batch_start(trainer, pl_module, batch, batch_idx)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Override to safely handle progress assertion"""
        if hasattr(self, 'progress') and self.progress is not None:
            super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_validation_end(self, trainer: Trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        sys.stdout.flush()
        if trainer.is_global_zero:
            metrics = trainer.logged_metrics
            infos = f"Epoch {trainer.current_epoch} metrics: "
            for k, v in metrics.items():
                if k.startswith('train/'):
                    continue
                value = v
                if isinstance(v, Tensor):
                    value = v.item()
                if isinstance(value, float):
                    infos += k + f"={value:.4f}  "
                else:
                    infos += k + f"={value}  "
            if len(metrics) > 0:
                sys.stdout.write(f'{infos}\n')
            sys.stdout.flush()
