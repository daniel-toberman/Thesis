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
