"""
ABI Framework - Training Example

Demonstrates the training API with progress tracking and checkpointing.
"""

import sys
import os

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import abi
from abi.training import (
    Trainer,
    TrainingConfig,
    train,
    TrainingMetrics,
)


def main():
    """Run training example."""
    print("ABI Training Example")
    print("=" * 40)

    # Initialize framework
    abi.init()

    # Example 1: Quick training with convenience function
    print("\n1. Quick Training")
    print("-" * 40)

    config = TrainingConfig(
        epochs=2,
        batch_size=32,
        learning_rate=0.001,
        optimizer="adamw",
    )

    print(f"Config: epochs={config.epochs}, batch_size={config.batch_size}, lr={config.learning_rate}")
    print("\nTraining...")

    report = train(config, verbose=True)

    print(f"\nResults:")
    print(f"  Final loss: {report.final_loss:.4f}")
    print(f"  Final accuracy: {report.final_accuracy:.4f}")
    print(f"  Total time: {report.total_time_seconds:.2f}s")

    # Example 2: Training with context manager for fine-grained control
    print("\n\n2. Training with Context Manager")
    print("-" * 40)

    config = TrainingConfig(
        epochs=3,
        batch_size=16,
        learning_rate=2e-5,
        optimizer="adamw",
        warmup_steps=50,
        gradient_clip_norm=1.0,
    )

    losses = []
    epoch_losses = []

    def on_epoch_end(epoch: int, avg_loss: float):
        epoch_losses.append((epoch, avg_loss))
        print(f"  Epoch {epoch} complete: avg_loss={avg_loss:.4f}")

    with Trainer(config) as trainer:
        print("Training with progress tracking...")

        for metrics in trainer.train(on_epoch=on_epoch_end):
            losses.append(metrics.loss)

            # Print progress every 50 steps
            if metrics.step % 50 == 0:
                print(f"  Step {metrics.step:4d}: loss={metrics.loss:.4f}, lr={metrics.learning_rate:.2e}")

        # Get final report
        report = trainer.get_report()

        print(f"\nFinal Report:")
        print(f"  Epochs completed: {report.epochs}")
        print(f"  Total batches: {report.batches}")
        print(f"  Best loss: {report.best_loss:.4f}")
        print(f"  Final loss: {report.final_loss:.4f}")

    # Example 3: Pre-configured training profiles
    print("\n\n3. Pre-configured Training Profiles")
    print("-" * 40)

    finetune = TrainingConfig.for_finetuning()
    print(f"Fine-tuning config:")
    print(f"  epochs={finetune.epochs}, lr={finetune.learning_rate}, batch_size={finetune.batch_size}")

    pretrain = TrainingConfig.for_pretraining()
    print(f"\nPre-training config:")
    print(f"  epochs={pretrain.epochs}, lr={pretrain.learning_rate}, batch_size={pretrain.batch_size}")
    print(f"  mixed_precision={pretrain.mixed_precision}, warmup_steps={pretrain.warmup_steps}")

    # Cleanup
    abi.shutdown()

    print("\n" + "=" * 40)
    print("Training example complete!")


if __name__ == "__main__":
    main()
