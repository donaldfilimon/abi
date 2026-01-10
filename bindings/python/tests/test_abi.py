"""Tests for ABI Python bindings."""

import pytest
import math
import sys
import os

# Add parent directory to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import abi


class TestCore:
    """Test core ABI functionality."""

    def test_version(self):
        """Test version retrieval."""
        version = abi.version()
        assert isinstance(version, str)
        assert len(version) > 0

    def test_init_shutdown(self):
        """Test initialization and shutdown."""
        abi.init()
        assert abi.is_initialized()
        abi.shutdown()


class TestVectorOperations:
    """Test vector math operations."""

    def test_cosine_similarity_identical(self):
        """Identical vectors should have similarity 1.0."""
        a = [1.0, 2.0, 3.0]
        b = [1.0, 2.0, 3.0]
        sim = abi.cosine_similarity(a, b)
        assert abs(sim - 1.0) < 1e-5

    def test_cosine_similarity_orthogonal(self):
        """Orthogonal vectors should have similarity 0.0."""
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        sim = abi.cosine_similarity(a, b)
        assert abs(sim) < 1e-5

    def test_cosine_similarity_opposite(self):
        """Opposite vectors should have similarity -1.0."""
        a = [1.0, 2.0, 3.0]
        b = [-1.0, -2.0, -3.0]
        sim = abi.cosine_similarity(a, b)
        assert abs(sim + 1.0) < 1e-5

    def test_vector_dot(self):
        """Test dot product."""
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        dot = abi.vector_dot(a, b)
        expected = 1 * 4 + 2 * 5 + 3 * 6
        assert abs(dot - expected) < 1e-5

    def test_vector_add(self):
        """Test vector addition."""
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        result = abi.vector_add(a, b)
        expected = [5.0, 7.0, 9.0]
        for i, (r, e) in enumerate(zip(result, expected)):
            assert abs(r - e) < 1e-5, f"Mismatch at index {i}"

    def test_l2_norm(self):
        """Test L2 norm."""
        vec = [3.0, 4.0]
        norm = abi.l2_norm(vec)
        assert abs(norm - 5.0) < 1e-5

    def test_l2_norm_unit(self):
        """Test L2 norm of unit vector."""
        vec = [1.0, 0.0, 0.0]
        norm = abi.l2_norm(vec)
        assert abs(norm - 1.0) < 1e-5

    def test_dimension_mismatch(self):
        """Test that dimension mismatch raises error."""
        a = [1.0, 2.0]
        b = [1.0, 2.0, 3.0]
        with pytest.raises(ValueError):
            abi.cosine_similarity(a, b)


class TestVectorDatabase:
    """Test vector database functionality."""

    def test_create_database(self):
        """Test database creation."""
        db = abi.VectorDatabase(name="test_db")
        assert db.count() == 0

    def test_add_vector(self):
        """Test adding vectors."""
        db = abi.VectorDatabase()
        vec_id = db.add([1.0, 2.0, 3.0])
        assert vec_id == 0
        assert db.count() == 1

    def test_add_with_metadata(self):
        """Test adding vectors with metadata."""
        db = abi.VectorDatabase()
        vec_id = db.add([1.0, 2.0, 3.0], metadata={"label": "test"})
        assert vec_id == 0

    def test_query(self):
        """Test querying vectors."""
        db = abi.VectorDatabase()
        db.add([1.0, 0.0, 0.0], metadata={"label": "x"})
        db.add([0.0, 1.0, 0.0], metadata={"label": "y"})
        db.add([0.0, 0.0, 1.0], metadata={"label": "z"})

        results = db.query([0.9, 0.1, 0.0], top_k=2)
        assert len(results) == 2
        # First result should be closest to x-axis
        assert results[0]["metadata"]["label"] == "x"

    def test_clear(self):
        """Test clearing database."""
        db = abi.VectorDatabase()
        db.add([1.0, 2.0, 3.0])
        db.add([4.0, 5.0, 6.0])
        assert db.count() == 2
        db.clear()
        assert db.count() == 0

    def test_dimension_consistency(self):
        """Test that dimension is enforced after first add."""
        db = abi.VectorDatabase()
        db.add([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            db.add([1.0, 2.0])  # Wrong dimension


class TestAgent:
    """Test AI agent functionality."""

    def test_create_agent(self):
        """Test agent creation."""
        agent = abi.Agent(name="test-agent")
        assert agent.name == "test-agent"

    def test_process_message(self):
        """Test processing a message."""
        agent = abi.Agent()
        response = agent.process("Hello")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_history(self):
        """Test conversation history."""
        agent = abi.Agent()
        agent.process("Hello")
        history = agent.get_history()
        assert len(history) == 2  # User message + assistant response
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_clear_history(self):
        """Test clearing history."""
        agent = abi.Agent()
        agent.process("Hello")
        agent.clear_history()
        assert len(agent.get_history()) == 0


class TestFeatures:
    """Test feature flags."""

    def test_feature_enum(self):
        """Test feature enumeration values."""
        assert abi.Feature.AI == 0
        assert abi.Feature.GPU == 1
        assert abi.Feature.DATABASE == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
