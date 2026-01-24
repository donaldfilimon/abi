"""Tests for streaming inference API."""

import pytest
import sys
import os
from typing import List

# Add parent directory to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from abi.llm import LlmEngine, InferenceConfig
from abi.llm_streaming import TokenEvent, StreamingConfig


class TestStreamingConfig:
    """Test streaming configuration."""

    def test_default_config(self):
        """Default config should have reasonable values."""
        config = StreamingConfig()
        assert config.max_tokens > 0
        assert config.temperature >= 0.0
        assert config.temperature <= 2.0

    def test_config_custom_values(self):
        """Config should accept custom values."""
        config = StreamingConfig(
            max_tokens=100,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.2,
            seed=42,
        )
        assert config.max_tokens == 100
        assert config.temperature == 0.8
        assert config.top_p == 0.95
        assert config.top_k == 50
        assert config.seed == 42

    def test_config_to_c(self):
        """Config should convert to C struct."""
        config = StreamingConfig(max_tokens=50, temperature=0.5)
        c_config = config.to_c()
        assert c_config.max_tokens == 50
        assert abs(c_config.temperature - 0.5) < 1e-5


class TestStreamingInference:
    """Test streaming inference."""

    @pytest.fixture
    def engine_with_mock_model(self, tmp_path):
        """Create engine with mock model file."""
        # Create a dummy model file
        model_path = tmp_path / "test_model.gguf"
        model_path.write_bytes(b"GGUF" + b"\x00" * 100)

        engine = LlmEngine()
        engine.load_model(str(model_path))
        return engine

    def test_generate_streaming_returns_iterator(self, engine_with_mock_model):
        """generate_streaming should return an iterator."""
        result = engine_with_mock_model.generate_streaming("Hello")
        # Should be iterable
        assert hasattr(result, "__iter__")

    def test_streaming_yields_tokens(self, engine_with_mock_model):
        """Streaming should yield string tokens."""
        tokens: List[str] = []

        for token in engine_with_mock_model.generate_streaming("Hello", max_tokens=10):
            tokens.append(token)
            if len(tokens) >= 5:
                break

        assert len(tokens) > 0
        for token in tokens:
            assert isinstance(token, str)

    def test_streaming_stop_early(self, engine_with_mock_model):
        """Should be able to stop streaming early."""
        count = 0

        for _ in engine_with_mock_model.generate_streaming("Test", max_tokens=100):
            count += 1
            if count >= 3:
                break

        assert count == 3

    def test_streaming_respects_max_tokens(self, engine_with_mock_model):
        """Streaming should respect max_tokens limit."""
        max_tokens = 5
        tokens = list(engine_with_mock_model.generate_streaming("Hello", max_tokens=max_tokens))
        assert len(tokens) <= max_tokens

    def test_streaming_with_callback(self, engine_with_mock_model):
        """Streaming should call callback for each token."""
        callback_tokens = []

        def callback(token):
            callback_tokens.append(token)

        tokens = list(engine_with_mock_model.generate_streaming(
            "Test",
            max_tokens=5,
            callback=callback
        ))

        assert len(callback_tokens) == len(tokens)
        for i, token in enumerate(tokens):
            assert callback_tokens[i] == token

    def test_streaming_without_model_raises(self):
        """Streaming without loaded model should raise."""
        engine = LlmEngine()
        with pytest.raises(RuntimeError, match="No model loaded"):
            list(engine.generate_streaming("Hello"))


class TestTokenEvent:
    """Test TokenEvent structure."""

    def test_token_event_fields(self):
        """TokenEvent should have required fields."""
        event = TokenEvent(
            token_id=42,
            text="hello",
            position=5,
            is_final=False,
            timestamp_ns=1234567890
        )
        assert event.text == "hello"
        assert event.token_id == 42
        assert event.position == 5
        assert event.is_final is False
        assert event.timestamp_ns == 1234567890

    def test_final_token(self):
        """Final token should be marked."""
        event = TokenEvent(
            token_id=0,
            text="",
            position=10,
            is_final=True,
            timestamp_ns=0
        )
        assert event.is_final is True

    def test_token_event_with_none_text(self):
        """TokenEvent can have None text."""
        event = TokenEvent(
            token_id=100,
            text=None,
            position=0,
            is_final=False,
            timestamp_ns=0
        )
        assert event.text is None


class TestInferenceConfig:
    """Test InferenceConfig for streaming compatibility."""

    def test_streaming_enabled(self):
        """Default config should have streaming enabled."""
        config = InferenceConfig.default()
        assert config.streaming is True

    def test_creative_config(self):
        """Creative config should have higher temperature."""
        config = InferenceConfig.creative()
        assert config.temperature == 1.0
        assert config.top_p == 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
