"""Tests for training pipeline API."""

import pytest
import os
import sys
import tempfile

# Add parent directory to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from abi.training import (
    TrainingConfig,
    TrainingReport,
    TrainingMetrics,
    Trainer,
    train,
    OptimizerType,
    LearningRateSchedule,
)


class TestTrainingConfig:
    """Test training configuration."""

    def test_default_config(self):
        """Default config should have reasonable values."""
        config = TrainingConfig.defaults()
        assert config.epochs > 0
        assert config.batch_size > 0
        assert config.learning_rate > 0

    def test_finetuning_config(self):
        """Fine-tuning config should have lower learning rate."""
        config = TrainingConfig.for_finetuning()
        assert config.learning_rate < 1e-4
        assert config.epochs <= 5

    def test_pretraining_config(self):
        """Pre-training config should have larger batch size."""
        config = TrainingConfig.for_pretraining()
        assert config.batch_size >= 128
        assert config.mixed_precision is True

    def test_config_custom_values(self):
        """Should accept custom values."""
        config = TrainingConfig(
            epochs=20,
            batch_size=64,
            learning_rate=0.01,
            optimizer="sgd",
        )
        assert config.epochs == 20
        assert config.batch_size == 64
        assert config.optimizer == "sgd"

    def test_config_with_checkpoint(self):
        """Config should accept checkpoint settings."""
        config = TrainingConfig(
            epochs=5,
            checkpoint_interval=100,
            checkpoint_path="/tmp/checkpoints",
        )
        assert config.checkpoint_interval == 100
        assert config.checkpoint_path == "/tmp/checkpoints"


class TestTrainer:
    """Test Trainer context manager."""

    def test_trainer_context_manager(self):
        """Trainer should work as context manager."""
        config = TrainingConfig(epochs=1, batch_size=2)
        with Trainer(config) as trainer:
            assert trainer is not None

    def test_trainer_train_yields_metrics(self):
        """train() should yield metrics."""
        config = TrainingConfig(epochs=1, batch_size=2)
        metrics_list = []

        with Trainer(config) as trainer:
            for metrics in trainer.train():
                metrics_list.append(metrics)
                assert isinstance(metrics.loss, float)
                assert isinstance(metrics.step, int)
                if len(metrics_list) >= 5:
                    break

        assert len(metrics_list) > 0

    def test_trainer_get_report(self):
        """get_report() should return training summary."""
        config = TrainingConfig(epochs=1)

        with Trainer(config) as trainer:
            for metrics in trainer.train():
                if metrics.step >= 10:
                    break
            report = trainer.get_report()

        assert isinstance(report, TrainingReport)
        assert report.final_loss is not None

    def test_trainer_early_stop(self):
        """Training can be stopped early."""
        config = TrainingConfig(epochs=10)
        steps = 0

        with Trainer(config) as trainer:
            for _ in trainer.train():
                steps += 1
                if steps >= 5:
                    break

        assert steps == 5

    def test_trainer_stop_method(self):
        """Trainer.stop() should halt training."""
        config = TrainingConfig(epochs=10)
        steps = 0

        with Trainer(config) as trainer:
            for _ in trainer.train():
                steps += 1
                if steps >= 5:
                    trainer.stop()
            # Verify training stopped
            assert not trainer.is_training

    def test_trainer_metrics_history(self):
        """Trainer should track metrics history."""
        config = TrainingConfig(epochs=1)

        with Trainer(config) as trainer:
            for metrics in trainer.train():
                if metrics.step >= 10:
                    break
            history = trainer.metrics_history

        assert len(history) > 0
        assert all(isinstance(m, TrainingMetrics) for m in history)

    def test_trainer_callbacks(self):
        """Trainer should call step callbacks."""
        config = TrainingConfig(epochs=1)
        step_calls = []

        def on_step(metrics):
            step_calls.append(metrics.step)

        with Trainer(config) as trainer:
            for metrics in trainer.train(on_step=on_step):
                if metrics.step >= 5:
                    break

        assert len(step_calls) > 0

    def test_trainer_not_initialized_raises(self):
        """Operations without context manager should raise."""
        config = TrainingConfig(epochs=1)
        trainer = Trainer(config)

        with pytest.raises(RuntimeError, match="not initialized"):
            trainer.get_report()


class TestTrainingMetrics:
    """Test TrainingMetrics structure."""

    def test_metrics_fields(self):
        """Metrics should have required fields."""
        metrics = TrainingMetrics(
            step=100,
            epoch=1,
            loss=0.5,
            learning_rate=0.001,
            accuracy=0.9,
            gradient_norm=0.3,
        )
        assert metrics.step == 100
        assert metrics.epoch == 1
        assert metrics.loss == 0.5
        assert metrics.accuracy == 0.9

    def test_metrics_repr(self):
        """Metrics should have readable repr."""
        metrics = TrainingMetrics(
            step=100,
            epoch=1,
            loss=0.5,
            learning_rate=0.001,
        )
        repr_str = repr(metrics)
        assert "step=100" in repr_str
        assert "loss=" in repr_str


class TestTrainingReport:
    """Test TrainingReport structure."""

    def test_report_fields(self):
        """Report should have summary fields."""
        report = TrainingReport(
            epochs=10,
            batches=1000,
            final_loss=0.01,
            final_accuracy=0.99,
            best_loss=0.008,
            gradient_updates=1000,
            checkpoints_saved=5,
            early_stopped=False,
            total_time_ms=3600000,
        )
        assert report.epochs == 10
        assert report.final_loss == 0.01
        assert report.best_loss == 0.008
        assert report.checkpoints_saved == 5

    def test_report_time_properties(self):
        """Report should compute time properties."""
        report = TrainingReport(
            total_time_ms=60000,  # 1 minute
        )
        assert report.total_time_seconds == 60.0
        assert report.total_time_minutes == 1.0

    def test_report_repr(self):
        """Report should have readable repr."""
        report = TrainingReport(
            epochs=5,
            batches=500,
            final_loss=0.05,
            final_accuracy=0.95,
        )
        repr_str = repr(report)
        assert "epochs=5" in repr_str
        assert "final_loss=" in repr_str


class TestConvenienceFunction:
    """Test train() convenience function."""

    def test_train_function(self):
        """train() function should work end-to-end."""
        # Use very short training for speed
        config = TrainingConfig(epochs=1)
        # Stop after a few steps by setting verbose=False to speed up
        report = train(config, verbose=False)

        assert isinstance(report, TrainingReport)


class TestCheckpoints:
    """Test checkpoint functionality."""

    def test_checkpoint_save(self):
        """Should save checkpoints when configured."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "checkpoint.ckpt")
            config = TrainingConfig(epochs=1)

            with Trainer(config) as trainer:
                for metrics in trainer.train():
                    if metrics.step >= 5:
                        trainer.save_checkpoint(checkpoint_path)
                        break

            # Check that checkpoint file was created (mock writes placeholder)
            assert os.path.exists(checkpoint_path)

    def test_checkpoint_without_trainer_raises(self):
        """Saving checkpoint without context manager should raise."""
        config = TrainingConfig(epochs=1)
        trainer = Trainer(config)

        with pytest.raises(RuntimeError, match="not initialized"):
            trainer.save_checkpoint("/tmp/test.ckpt")


class TestOptimizerTypes:
    """Test optimizer type enumeration."""

    def test_optimizer_values(self):
        """OptimizerType should have expected values."""
        assert OptimizerType.SGD.value == "sgd"
        assert OptimizerType.ADAM.value == "adam"
        assert OptimizerType.ADAMW.value == "adamw"


class TestLearningRateSchedule:
    """Test learning rate schedule enumeration."""

    def test_schedule_types(self):
        """LearningRateSchedule should have expected types."""
        assert LearningRateSchedule.CONSTANT is not None
        assert LearningRateSchedule.COSINE is not None
        assert LearningRateSchedule.WARMUP_COSINE is not None
        assert LearningRateSchedule.STEP is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
