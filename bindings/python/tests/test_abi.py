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

    def test_feature_enum(self):
        """Test feature enumeration values."""
        assert abi.Feature.AI == 0
        assert abi.Feature.GPU == 1
        assert abi.Feature.DATABASE == 3
        assert abi.Feature.LLM == 7


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
        assert db.count == 0

    def test_add_vector(self):
        """Test adding vectors."""
        db = abi.VectorDatabase()
        vec_id = db.add([1.0, 2.0, 3.0])
        assert vec_id == 0
        assert db.count == 1

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

        results = db.search([0.9, 0.1, 0.0], top_k=2)
        assert len(results) == 2
        # First result should be closest to x-axis
        assert results[0].metadata["label"] == "x"

    def test_clear(self):
        """Test clearing database."""
        db = abi.VectorDatabase()
        db.add([1.0, 2.0, 3.0])
        db.add([4.0, 5.0, 6.0])
        assert db.count == 2
        db.clear()
        assert db.count == 0

    def test_dimension_consistency(self):
        """Test that dimension is enforced after first add."""
        db = abi.VectorDatabase()
        db.add([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            db.add([1.0, 2.0])  # Wrong dimension

    def test_batch_add(self):
        """Test batch adding vectors."""
        db = abi.VectorDatabase()
        result = db.add_batch([
            {"vector": [1.0, 2.0, 3.0], "metadata": {"label": "a"}},
            {"vector": [4.0, 5.0, 6.0], "metadata": {"label": "b"}},
        ])
        assert result.success_count == 2
        assert result.error_count == 0
        assert db.count == 2

    def test_get_vector(self):
        """Test getting vector by ID."""
        db = abi.VectorDatabase()
        db.add([1.0, 2.0, 3.0], metadata={"label": "test"})
        vec = db.get(0)
        assert vec is not None
        assert vec["metadata"]["label"] == "test"

    def test_delete_vector(self):
        """Test deleting vector."""
        db = abi.VectorDatabase()
        db.add([1.0, 2.0, 3.0])
        assert db.count == 1
        result = db.delete(0)
        assert result is True
        assert db.count == 0

    def test_stats(self):
        """Test database statistics."""
        db = abi.VectorDatabase(dimensions=3)
        db.add([1.0, 2.0, 3.0])
        db.add([4.0, 5.0, 6.0])
        stats = db.stats()
        assert stats.vector_count == 2
        assert stats.dimensions == 3


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


class TestConfiguration:
    """Test configuration functionality."""

    def test_config_defaults(self):
        """Test default configuration."""
        from abi.config import Config
        config = Config.defaults()
        assert config.gpu is not None
        assert config.ai is not None
        assert config.database is not None

    def test_config_minimal(self):
        """Test minimal configuration."""
        from abi.config import Config
        config = Config.minimal()
        assert config.gpu is None
        assert config.ai is None
        assert config.database is None

    def test_config_is_enabled(self):
        """Test feature checking."""
        from abi.config import Config, GpuConfig
        config = Config(gpu=GpuConfig())
        assert config.is_enabled("gpu")
        assert not config.is_enabled("database")

    def test_config_builder(self):
        """Test configuration builder."""
        from abi.config import ConfigBuilder, GpuConfig
        config = (ConfigBuilder()
            .with_gpu(GpuConfig())
            .build())
        assert config.gpu is not None

    def test_gpu_config_backends(self):
        """Test GPU configuration backends."""
        from abi.config import GpuConfig, GpuBackend
        cuda = GpuConfig.cuda()
        assert cuda.backend == GpuBackend.CUDA

        vulkan = GpuConfig.vulkan()
        assert vulkan.backend == GpuBackend.VULKAN

        cpu = GpuConfig.cpu_only()
        assert cpu.backend == GpuBackend.CPU

    def test_llm_config(self):
        """Test LLM configuration."""
        from abi.config import LlmConfig
        config = LlmConfig(
            model_path="./model.gguf",
            context_size=4096,
            use_gpu=True,
        )
        assert config.model_path == "./model.gguf"
        assert config.context_size == 4096
        assert config.use_gpu is True

    def test_database_config(self):
        """Test database configuration."""
        from abi.config import DatabaseConfig, IndexType, DistanceMetric
        config = DatabaseConfig(
            path="./vectors.db",
            index_type=IndexType.HNSW,
            distance_metric=DistanceMetric.COSINE,
        )
        assert config.path == "./vectors.db"
        assert config.index_type == IndexType.HNSW
        assert config.distance_metric == DistanceMetric.COSINE

    def test_database_config_in_memory(self):
        """Test in-memory database configuration."""
        from abi.config import DatabaseConfig
        config = DatabaseConfig.in_memory()
        assert config.path == ":memory:"
        assert config.wal_enabled is False


class TestLLM:
    """Test LLM functionality."""

    def test_inference_config(self):
        """Test inference configuration."""
        from abi.llm import InferenceConfig
        config = InferenceConfig(
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=256,
        )
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.max_new_tokens == 256

    def test_inference_config_greedy(self):
        """Test greedy sampling configuration."""
        from abi.llm import InferenceConfig
        config = InferenceConfig.greedy()
        assert config.temperature == 0.0
        assert config.top_k == 1

    def test_llm_engine_creation(self):
        """Test LLM engine creation."""
        from abi.llm import LlmEngine
        engine = LlmEngine()
        assert not engine.is_loaded
        assert engine.model_info is None

    def test_inference_stats(self):
        """Test inference statistics."""
        from abi.llm import InferenceStats
        stats = InferenceStats(
            prefill_time_ns=1_000_000_000,
            decode_time_ns=2_000_000_000,
            prompt_tokens=100,
            generated_tokens=50,
        )
        assert stats.prefill_tokens_per_second == 100.0
        assert stats.decode_tokens_per_second == 25.0
        assert stats.total_time_seconds == 3.0


class TestGPU:
    """Test GPU functionality."""

    def test_gpu_context_creation(self):
        """Test GPU context creation."""
        from abi.gpu import GpuContext
        ctx = GpuContext()
        # May or may not have GPU available
        assert isinstance(ctx.is_available, bool)

    def test_gpu_config(self):
        """Test GPU configuration."""
        from abi.gpu import GpuConfig, GpuBackend
        config = GpuConfig(
            backend=GpuBackend.AUTO,
            device_index=0,
            memory_limit=4 * 1024 ** 3,
        )
        assert config.backend == GpuBackend.AUTO
        assert config.device_index == 0
        assert config.memory_limit == 4 * 1024 ** 3

    def test_list_devices(self):
        """Test listing GPU devices."""
        from abi.gpu import GpuContext
        devices = GpuContext.list_devices()
        assert isinstance(devices, list)
        # Should at least have CPU fallback
        assert len(devices) > 0

    def test_gpu_operations(self):
        """Test GPU operations with fallback."""
        from abi.gpu import GpuContext
        ctx = GpuContext()

        # Vector add
        result = ctx.vector_add([1.0, 2.0], [3.0, 4.0])
        assert result == [4.0, 6.0]

        # Dot product
        dot = ctx.vector_dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        assert dot == 32.0

    def test_gpu_stats(self):
        """Test GPU statistics."""
        from abi.gpu import GpuContext
        ctx = GpuContext()
        ctx.vector_add([1.0], [2.0])
        stats = ctx.stats
        assert stats.total_ops >= 1


class TestDatabaseModule:
    """Test database module functionality."""

    def test_create_database_function(self):
        """Test create_database convenience function."""
        db = abi.create_database(name="test", dimensions=128)
        assert db.name == "test"
        assert db.dimensions == 128

    def test_filtered_search(self):
        """Test filtered search."""
        from abi.database import VectorDatabase
        db = VectorDatabase(dimensions=3)
        db.add([1.0, 0.0, 0.0], metadata={"category": "a", "year": 2023})
        db.add([0.0, 1.0, 0.0], metadata={"category": "b", "year": 2024})
        db.add([0.0, 0.0, 1.0], metadata={"category": "a", "year": 2024})

        # Filter by category
        results = db.search([1.0, 0.0, 0.0], top_k=10, filter={"category": "a"})
        assert len(results) == 2

        # Filter by year
        results = db.search([1.0, 0.0, 0.0], top_k=10, filter={"year": {"$gte": 2024}})
        assert len(results) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
