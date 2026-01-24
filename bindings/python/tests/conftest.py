"""Pytest configuration and fixtures."""

import pytest
import sys
import os

# Add parent directory to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def temp_db():
    """Create a temporary vector database."""
    from abi import VectorDatabase
    db = VectorDatabase(name="test_temp")
    yield db
    db.clear()


@pytest.fixture
def sample_vectors():
    """Provide sample vectors for testing."""
    return [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.707, 0.707, 0.0],
        [0.577, 0.577, 0.577],
    ]


@pytest.fixture
def training_config():
    """Provide default training config."""
    from abi.training import TrainingConfig
    return TrainingConfig(epochs=1, batch_size=2)


@pytest.fixture
def streaming_config():
    """Provide default streaming config."""
    from abi.llm_streaming import StreamingConfig
    return StreamingConfig(max_tokens=10)


@pytest.fixture
def mock_model_file(tmp_path):
    """Create a mock GGUF model file."""
    model_path = tmp_path / "test_model.gguf"
    model_path.write_bytes(b"GGUF" + b"\x00" * 100)
    return str(model_path)


@pytest.fixture
def llm_engine(mock_model_file):
    """Create LLM engine with mock model loaded."""
    from abi.llm import LlmEngine
    engine = LlmEngine()
    engine.load_model(mock_model_file)
    yield engine
    engine.unload_model()
