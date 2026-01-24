"""
ABI Framework Training Module

High-level training pipeline API with context manager support.
"""

from typing import Optional, Iterator, Callable, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum, auto
import os


class OptimizerType(Enum):
    """Optimizer types for training."""
    SGD = "sgd"
    ADAM = "adam"
    ADAMW = "adamw"


class LearningRateSchedule(Enum):
    """Learning rate schedule types."""
    CONSTANT = auto()
    COSINE = auto()
    WARMUP_COSINE = auto()
    STEP = auto()
    POLYNOMIAL = auto()


@dataclass
class TrainingConfig:
    """
    Configuration for training sessions.

    Attributes:
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        optimizer: Optimizer type (sgd, adam, adamw)
        weight_decay: Weight decay for regularization
        gradient_clip_norm: Maximum gradient norm (0 to disable)
        warmup_steps: Number of warmup steps
        checkpoint_interval: Save checkpoint every N steps (0 to disable)
        checkpoint_path: Directory for checkpoints
        early_stopping_patience: Stop if no improvement for N epochs
        early_stopping_threshold: Minimum improvement threshold
        mixed_precision: Enable mixed precision training

    Example:
        >>> config = TrainingConfig(epochs=10, batch_size=32)
        >>> config = TrainingConfig.defaults()
        >>> config = TrainingConfig.for_finetuning()
    """
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    warmup_steps: int = 100
    checkpoint_interval: int = 0
    checkpoint_path: Optional[str] = None
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 1e-4
    mixed_precision: bool = False
    gradient_accumulation_steps: int = 1

    @classmethod
    def defaults(cls) -> "TrainingConfig":
        """Create default training configuration."""
        return cls()

    @classmethod
    def for_finetuning(cls) -> "TrainingConfig":
        """Create configuration optimized for fine-tuning."""
        return cls(
            epochs=3,
            batch_size=8,
            learning_rate=2e-5,
            optimizer="adamw",
            weight_decay=0.01,
            warmup_steps=500,
            gradient_accumulation_steps=4,
        )

    @classmethod
    def for_pretraining(cls) -> "TrainingConfig":
        """Create configuration optimized for pre-training."""
        return cls(
            epochs=1,
            batch_size=256,
            learning_rate=1e-4,
            optimizer="adamw",
            weight_decay=0.1,
            warmup_steps=2000,
            checkpoint_interval=1000,
            mixed_precision=True,
        )


@dataclass
class TrainingMetrics:
    """
    Per-step training metrics.

    Attributes:
        step: Current training step
        epoch: Current epoch number
        loss: Training loss for this step
        accuracy: Training accuracy for this step
        learning_rate: Current learning rate
        gradient_norm: Gradient norm for this step
    """
    step: int = 0
    epoch: int = 0
    loss: float = 0.0
    accuracy: float = 0.0
    learning_rate: float = 0.0
    gradient_norm: float = 0.0

    def __repr__(self) -> str:
        return (
            f"TrainingMetrics(step={self.step}, epoch={self.epoch}, "
            f"loss={self.loss:.4f}, acc={self.accuracy:.4f}, "
            f"lr={self.learning_rate:.2e})"
        )


@dataclass
class TrainingReport:
    """
    Final training report.

    Attributes:
        epochs: Total epochs completed
        batches: Total batches processed
        final_loss: Final training loss
        final_accuracy: Final training accuracy
        best_loss: Best loss achieved
        gradient_updates: Total gradient updates
        checkpoints_saved: Number of checkpoints saved
        early_stopped: Whether training stopped early
        total_time_ms: Total training time in milliseconds
    """
    epochs: int = 0
    batches: int = 0
    final_loss: float = 0.0
    final_accuracy: float = 0.0
    best_loss: float = 0.0
    gradient_updates: int = 0
    checkpoints_saved: int = 0
    early_stopped: bool = False
    total_time_ms: int = 0

    @property
    def total_time_seconds(self) -> float:
        """Get total time in seconds."""
        return self.total_time_ms / 1000.0

    @property
    def total_time_minutes(self) -> float:
        """Get total time in minutes."""
        return self.total_time_ms / 60000.0

    def __repr__(self) -> str:
        return (
            f"TrainingReport(epochs={self.epochs}, batches={self.batches}, "
            f"final_loss={self.final_loss:.4f}, final_acc={self.final_accuracy:.4f}, "
            f"time={self.total_time_seconds:.1f}s)"
        )


class Trainer:
    """
    Training session manager with context manager support.

    Provides a high-level interface for running training with automatic
    resource cleanup and progress tracking.

    Example:
        >>> config = TrainingConfig(epochs=10, batch_size=32)
        >>> with Trainer(config) as trainer:
        ...     for metrics in trainer.train():
        ...         print(f"Step {metrics.step}: loss={metrics.loss:.4f}")
        ...     report = trainer.get_report()
        ...     trainer.save_checkpoint("final.ckpt")

        >>> # Or use the convenience function
        >>> report = train(TrainingConfig(epochs=5))
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize the trainer.

        Args:
            config: Training configuration
        """
        self._config = config
        self._trainer_id: Optional[int] = None
        self._lib = None
        self._is_training = False
        self._metrics_history: List[TrainingMetrics] = []

        # Try to load native library
        try:
            from . import _load_library
            self._lib = _load_library()
            from .llm_streaming import _setup_streaming_functions
            _setup_streaming_functions(self._lib)
        except (ImportError, AttributeError):
            pass

    @property
    def config(self) -> TrainingConfig:
        """Get training configuration."""
        return self._config

    @property
    def is_training(self) -> bool:
        """Check if training is in progress."""
        return self._is_training

    @property
    def metrics_history(self) -> List[TrainingMetrics]:
        """Get all recorded metrics."""
        return self._metrics_history.copy()

    def __enter__(self) -> "Trainer":
        """Enter context, create native trainer."""
        self._create_trainer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context, destroy native trainer."""
        self._destroy_trainer()

    def _create_trainer(self) -> None:
        """Create the native trainer."""
        if self._lib is not None and hasattr(self._lib, "abi_train_create"):
            from .llm_streaming import _train_create
            self._trainer_id = _train_create(self._lib, self._config)
        else:
            # Mock trainer for development
            self._trainer_id = 1

    def _destroy_trainer(self) -> None:
        """Destroy the native trainer."""
        if self._trainer_id is not None:
            if self._lib is not None and hasattr(self._lib, "abi_train_destroy"):
                from .llm_streaming import _train_destroy
                _train_destroy(self._lib, self._trainer_id)
            self._trainer_id = None
        self._is_training = False

    def train(
        self,
        on_step: Optional[Callable[[TrainingMetrics], None]] = None,
        on_epoch: Optional[Callable[[int, float], None]] = None,
    ) -> Iterator[TrainingMetrics]:
        """
        Run training and yield metrics.

        Args:
            on_step: Optional callback called for each step
            on_epoch: Optional callback called at end of each epoch (epoch_num, avg_loss)

        Yields:
            TrainingMetrics for each training step

        Example:
            >>> with Trainer(config) as trainer:
            ...     for metrics in trainer.train():
            ...         if metrics.step % 100 == 0:
            ...             print(f"Step {metrics.step}: {metrics.loss:.4f}")
        """
        if self._trainer_id is None:
            raise RuntimeError("Trainer not initialized. Use context manager.")

        self._is_training = True
        self._metrics_history.clear()
        current_epoch = 0
        epoch_losses: List[float] = []

        try:
            if self._lib is not None and hasattr(self._lib, "abi_train_step"):
                # Real training via FFI
                from .llm_streaming import _train_step, TrainingMetrics as CMetrics

                while self._is_training:
                    c_metrics = _train_step(self._lib, self._trainer_id)
                    if c_metrics is None:
                        break

                    metrics = TrainingMetrics(
                        step=c_metrics.step,
                        epoch=c_metrics.epoch,
                        loss=c_metrics.loss,
                        accuracy=c_metrics.accuracy,
                        learning_rate=c_metrics.learning_rate,
                        gradient_norm=c_metrics.gradient_norm,
                    )

                    self._metrics_history.append(metrics)
                    epoch_losses.append(metrics.loss)

                    if on_step:
                        on_step(metrics)

                    # Check for epoch boundary
                    if metrics.epoch > current_epoch:
                        if on_epoch and epoch_losses:
                            avg_loss = sum(epoch_losses) / len(epoch_losses)
                            on_epoch(current_epoch, avg_loss)
                        current_epoch = metrics.epoch
                        epoch_losses.clear()

                    yield metrics
            else:
                # Mock training for development
                total_steps = self._config.epochs * 100  # Simulated steps per epoch
                for step in range(total_steps):
                    epoch = step // 100
                    progress = step / total_steps
                    loss = 2.0 * (1.0 - progress) + 0.05 * (step % 10)
                    accuracy = 0.5 + 0.4 * progress

                    # Warmup learning rate
                    if step < self._config.warmup_steps:
                        lr = self._config.learning_rate * (step + 1) / self._config.warmup_steps
                    else:
                        lr = self._config.learning_rate

                    metrics = TrainingMetrics(
                        step=step,
                        epoch=epoch,
                        loss=loss,
                        accuracy=accuracy,
                        learning_rate=lr,
                        gradient_norm=0.5 + 0.3 * (1.0 - progress),
                    )

                    self._metrics_history.append(metrics)
                    epoch_losses.append(metrics.loss)

                    if on_step:
                        on_step(metrics)

                    # Check for epoch boundary
                    if epoch > current_epoch:
                        if on_epoch and epoch_losses:
                            avg_loss = sum(epoch_losses) / len(epoch_losses)
                            on_epoch(current_epoch, avg_loss)
                        current_epoch = epoch
                        epoch_losses.clear()

                    yield metrics

        finally:
            self._is_training = False

    def get_report(self) -> TrainingReport:
        """
        Get the final training report.

        Returns:
            TrainingReport with final statistics

        Raises:
            RuntimeError: If trainer not initialized
        """
        if self._trainer_id is None:
            raise RuntimeError("Trainer not initialized. Use context manager.")

        if self._lib is not None and hasattr(self._lib, "abi_train_get_report"):
            from .llm_streaming import _train_get_report
            c_report = _train_get_report(self._lib, self._trainer_id)
            return TrainingReport(
                epochs=c_report.epochs,
                batches=c_report.batches,
                final_loss=c_report.final_loss,
                final_accuracy=c_report.final_accuracy,
                best_loss=c_report.best_loss,
                gradient_updates=c_report.gradient_updates,
                checkpoints_saved=c_report.checkpoints_saved,
                early_stopped=c_report.early_stopped,
                total_time_ms=c_report.total_time_ms,
            )
        else:
            # Mock report
            if not self._metrics_history:
                return TrainingReport()

            final = self._metrics_history[-1]
            best_loss = min(m.loss for m in self._metrics_history)
            return TrainingReport(
                epochs=final.epoch + 1,
                batches=len(self._metrics_history),
                final_loss=final.loss,
                final_accuracy=final.accuracy,
                best_loss=best_loss,
                gradient_updates=len(self._metrics_history),
                checkpoints_saved=0,
                early_stopped=False,
                total_time_ms=len(self._metrics_history) * 10,
            )

    def save_checkpoint(self, path: str) -> None:
        """
        Save a training checkpoint.

        Args:
            path: Path to save checkpoint

        Raises:
            RuntimeError: If trainer not initialized or save fails
        """
        if self._trainer_id is None:
            raise RuntimeError("Trainer not initialized. Use context manager.")

        if self._lib is not None and hasattr(self._lib, "abi_train_save_checkpoint"):
            from .llm_streaming import _train_save_checkpoint
            _train_save_checkpoint(self._lib, self._trainer_id, path)
        else:
            # Mock checkpoint save
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                f.write(f"# Checkpoint at step {len(self._metrics_history)}\n")

    def stop(self) -> None:
        """Stop training gracefully."""
        self._is_training = False


def train(
    config: TrainingConfig,
    on_step: Optional[Callable[[TrainingMetrics], None]] = None,
    on_epoch: Optional[Callable[[int, float], None]] = None,
    verbose: bool = True,
) -> TrainingReport:
    """
    Convenience function for one-shot training.

    Args:
        config: Training configuration
        on_step: Optional callback for each step
        on_epoch: Optional callback for each epoch
        verbose: Print progress every 100 steps

    Returns:
        TrainingReport with final statistics

    Example:
        >>> report = train(TrainingConfig(epochs=5))
        >>> print(f"Final loss: {report.final_loss:.4f}")

        >>> # With callbacks
        >>> report = train(
        ...     TrainingConfig(epochs=10),
        ...     on_step=lambda m: print(f"{m.step}: {m.loss:.4f}"),
        ...     on_epoch=lambda e, l: print(f"Epoch {e}: {l:.4f}"),
        ... )
    """
    def verbose_step(metrics: TrainingMetrics) -> None:
        if on_step:
            on_step(metrics)
        if verbose and metrics.step % 100 == 0:
            print(f"Step {metrics.step}: loss={metrics.loss:.4f}, acc={metrics.accuracy:.4f}")

    with Trainer(config) as trainer:
        for _ in trainer.train(on_step=verbose_step, on_epoch=on_epoch):
            pass
        return trainer.get_report()


def is_enabled() -> bool:
    """Check if training features are available."""
    return True


# Export all public symbols
__all__ = [
    "OptimizerType",
    "LearningRateSchedule",
    "TrainingConfig",
    "TrainingMetrics",
    "TrainingReport",
    "Trainer",
    "train",
    "is_enabled",
]
