"""Integration tests that require native library."""

import pytest
import os
import sys

# Add parent directory to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def has_native_library():
    """Check if native ABI library is available."""
    try:
        from abi import _load_library
        lib = _load_library()
        return lib is not None
    except (ImportError, OSError, AttributeError):
        return False


native_available = has_native_library()
skip_without_native = pytest.mark.skipif(
    not native_available,
    reason="Native library not available"
)


@skip_without_native
class TestNativeStreaming:
    """Tests requiring native library for streaming."""

    def test_native_streaming_basic(self, tmp_path):
        """Test native streaming with real library."""
        from abi.llm import LlmEngine
        from abi.llm_streaming import StreamingConfig

        # Create mock model file for engine
        model_path = tmp_path / "test_model.gguf"
        model_path.write_bytes(b"GGUF" + b"\x00" * 100)

        engine = LlmEngine()
        engine.load_model(str(model_path))
        tokens = []

        for token in engine.generate_streaming("Hello", max_tokens=5):
            tokens.append(token)

        # With native library, should get real tokens
        assert len(tokens) > 0
        for t in tokens:
            assert isinstance(t, str)

    def test_native_streaming_temperature(self, tmp_path):
        """Test that temperature parameter is accepted."""
        from abi.llm import LlmEngine

        # Create mock model file
        model_path = tmp_path / "test_model.gguf"
        model_path.write_bytes(b"GGUF" + b"\x00" * 100)

        engine = LlmEngine()
        engine.load_model(str(model_path))

        # Low temperature should work
        for token in engine.generate_streaming("The", max_tokens=3, temperature=0.1):
            break  # Just verify it starts

        # High temperature should work
        for token in engine.generate_streaming("The", max_tokens=3, temperature=1.5):
            break


@skip_without_native
class TestNativeTraining:
    """Tests requiring native library for training."""

    def test_native_training_basic(self):
        """Test native training with real library."""
        from abi.training import Trainer, TrainingConfig

        config = TrainingConfig(epochs=1, batch_size=2)

        with Trainer(config) as trainer:
            step_count = 0
            for metrics in trainer.train():
                step_count += 1
                assert metrics.loss >= 0
                if step_count >= 3:
                    break

            assert step_count > 0


class TestMockFallback:
    """Tests verifying mock fallback works."""

    def test_streaming_mock_fallback(self, tmp_path):
        """Streaming should work with mock when native unavailable."""
        from abi.llm import LlmEngine

        # Create mock model file
        model_path = tmp_path / "test_model.gguf"
        model_path.write_bytes(b"GGUF" + b"\x00" * 100)

        engine = LlmEngine()
        engine.load_model(str(model_path))
        tokens = list(engine.generate_streaming("Test", max_tokens=5))

        # Mock or native, should get tokens
        assert len(tokens) > 0

    def test_training_mock_fallback(self):
        """Training should work with mock when native unavailable."""
        from abi.training import Trainer, TrainingConfig

        config = TrainingConfig(epochs=1)

        with Trainer(config) as trainer:
            for metrics in trainer.train():
                assert hasattr(metrics, 'loss')
                break


class TestVectorDatabase:
    """Test vector database integration."""

    def test_database_add_and_search(self):
        """Database should work end-to-end."""
        import abi

        db = abi.VectorDatabase(name="test_integration")
        db.add([1.0, 0.0, 0.0], metadata={"label": "x"})
        db.add([0.0, 1.0, 0.0], metadata={"label": "y"})
        db.add([0.0, 0.0, 1.0], metadata={"label": "z"})

        results = db.search([0.9, 0.1, 0.0], top_k=1)

        assert len(results) == 1
        assert results[0].metadata["label"] == "x"

    def test_database_batch_operations(self):
        """Batch operations should work."""
        import abi

        db = abi.VectorDatabase()
        result = db.add_batch([
            {"vector": [1.0, 2.0, 3.0], "metadata": {"id": 1}},
            {"vector": [4.0, 5.0, 6.0], "metadata": {"id": 2}},
            {"vector": [7.0, 8.0, 9.0], "metadata": {"id": 3}},
        ])

        assert result.success_count == 3
        assert db.count == 3


class TestConfiguration:
    """Test configuration integration."""

    def test_config_builder(self):
        """Configuration builder should work."""
        from abi.config import ConfigBuilder, GpuConfig

        config = (ConfigBuilder()
            .with_gpu(GpuConfig())
            .build())

        assert config.gpu is not None

    def test_llm_config(self):
        """LLM configuration should be valid."""
        from abi.config import LlmConfig

        config = LlmConfig(
            model_path="./model.gguf",
            context_size=4096,
            use_gpu=True,
        )

        assert config.model_path == "./model.gguf"
        assert config.context_size == 4096


class TestAgent:
    """Test agent integration."""

    def test_agent_creation(self):
        """Agent should be creatable."""
        import abi

        agent = abi.Agent(name="test-agent")
        assert agent.name == "test-agent"

    def test_agent_process(self):
        """Agent should process messages."""
        import abi

        agent = abi.Agent()
        response = agent.process("Hello")

        assert isinstance(response, str)
        assert len(response) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
