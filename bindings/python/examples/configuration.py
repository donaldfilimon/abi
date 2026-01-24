#!/usr/bin/env python3
"""
ABI Framework Configuration Example

Demonstrates configuration options for GPU, AI, LLM,
database, and other framework features.
"""

import sys
import os

# Add parent directory to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import abi
from abi.config import (
    Config,
    ConfigBuilder,
    GpuConfig,
    GpuBackend,
    AiConfig,
    LlmConfig,
    EmbeddingsConfig,
    AgentsConfig,
    TrainingConfig,
    DatabaseConfig,
    IndexType,
    DistanceMetric,
    NetworkConfig,
    ObservabilityConfig,
    Optimizer,
)


def main():
    print("=" * 60)
    print("ABI Framework Configuration Example")
    print("=" * 60)

    # Basic configuration
    print("\n1. Basic Configuration")
    print("-" * 40)

    # Minimal configuration (all features disabled)
    minimal = Config.minimal()
    print(f"   Minimal config enabled features: {minimal.enabled_features()}")

    # Default configuration (all features enabled with defaults)
    defaults = Config.defaults()
    print(f"   Default config enabled features: {defaults.enabled_features()}")

    # GPU Configuration
    print("\n2. GPU Configuration")
    print("-" * 40)

    # Auto-detect backend
    gpu_auto = GpuConfig.defaults()
    print(f"   Auto-detect: backend={gpu_auto.backend.name}")

    # Specific backends
    gpu_cuda = GpuConfig.cuda(device_index=0)
    print(f"   CUDA: backend={gpu_cuda.backend.name}, device={gpu_cuda.device_index}")

    gpu_vulkan = GpuConfig.vulkan()
    print(f"   Vulkan: backend={gpu_vulkan.backend.name}")

    gpu_metal = GpuConfig.metal()
    print(f"   Metal: backend={gpu_metal.backend.name}")

    # CPU only (disable GPU)
    gpu_cpu = GpuConfig.cpu_only()
    print(f"   CPU only: backend={gpu_cpu.backend.name}")

    # Custom GPU config
    gpu_custom = GpuConfig(
        backend=GpuBackend.CUDA,
        device_index=0,
        memory_limit=8 * 1024 ** 3,  # 8GB
        async_enabled=True,
        cache_kernels=True,
    )
    print(f"   Custom: memory_limit={gpu_custom.memory_limit / (1024**3):.0f}GB")

    # AI Configuration
    print("\n3. AI Configuration")
    print("-" * 40)

    # Full AI config
    ai_full = AiConfig.defaults()
    print(f"   Full AI config:")
    print(f"      LLM: {ai_full.llm is not None}")
    print(f"      Embeddings: {ai_full.embeddings is not None}")
    print(f"      Agents: {ai_full.agents is not None}")

    # LLM only
    ai_llm = AiConfig.llm_only(LlmConfig(
        model_path="./models/llama.gguf",
        context_size=4096,
        use_gpu=True,
    ))
    print(f"   LLM only config:")
    print(f"      Model path: {ai_llm.llm.model_path}")
    print(f"      Context size: {ai_llm.llm.context_size}")

    # Embeddings only
    ai_embed = AiConfig.embeddings_only(EmbeddingsConfig(
        model="sentence-transformers",
        dimension=768,
        normalize=True,
    ))
    print(f"   Embeddings only config:")
    print(f"      Model: {ai_embed.embeddings.model}")
    print(f"      Dimension: {ai_embed.embeddings.dimension}")

    # LLM Configuration
    print("\n4. LLM Configuration")
    print("-" * 40)

    llm_config = LlmConfig(
        model_path="./models/mistral-7b-q4.gguf",
        model_name="mistral",
        context_size=8192,
        threads=8,
        use_gpu=True,
        batch_size=1024,
    )
    print(f"   Model: {llm_config.model_name}")
    print(f"   Path: {llm_config.model_path}")
    print(f"   Context: {llm_config.context_size}")
    print(f"   Threads: {llm_config.threads}")
    print(f"   GPU: {llm_config.use_gpu}")
    print(f"   Batch size: {llm_config.batch_size}")

    # Database Configuration
    print("\n5. Database Configuration")
    print("-" * 40)

    # File-based database
    db_file = DatabaseConfig(
        path="./data/vectors.db",
        index_type=IndexType.HNSW,
        distance_metric=DistanceMetric.COSINE,
        cache_size=128 * 1024 * 1024,  # 128MB
        wal_enabled=True,
    )
    print(f"   File-based:")
    print(f"      Path: {db_file.path}")
    print(f"      Index: {db_file.index_type.name}")
    print(f"      Metric: {db_file.distance_metric.name}")
    print(f"      Cache: {db_file.cache_size / (1024*1024):.0f}MB")

    # In-memory database
    db_memory = DatabaseConfig.in_memory()
    print(f"   In-memory:")
    print(f"      Path: {db_memory.path}")
    print(f"      WAL: {db_memory.wal_enabled}")

    # Training Configuration
    print("\n6. Training Configuration")
    print("-" * 40)

    training = TrainingConfig(
        epochs=10,
        batch_size=32,
        learning_rate=0.001,
        optimizer=Optimizer.ADAMW,
        checkpoint_dir="./checkpoints",
        checkpoint_frequency=2,
    )
    print(f"   Epochs: {training.epochs}")
    print(f"   Batch size: {training.batch_size}")
    print(f"   Learning rate: {training.learning_rate}")
    print(f"   Optimizer: {training.optimizer.name}")
    print(f"   Checkpoint dir: {training.checkpoint_dir}")

    # Network Configuration
    print("\n7. Network Configuration")
    print("-" * 40)

    network = NetworkConfig(
        host="0.0.0.0",
        port=8080,
        cluster_enabled=True,
        max_connections=1000,
    )
    print(f"   Host: {network.host}")
    print(f"   Port: {network.port}")
    print(f"   Cluster: {network.cluster_enabled}")
    print(f"   Max connections: {network.max_connections}")

    # Observability Configuration
    print("\n8. Observability Configuration")
    print("-" * 40)

    observability = ObservabilityConfig(
        metrics_enabled=True,
        tracing_enabled=True,
        logging_level="debug",
    )
    print(f"   Metrics: {observability.metrics_enabled}")
    print(f"   Tracing: {observability.tracing_enabled}")
    print(f"   Log level: {observability.logging_level}")

    # Using ConfigBuilder
    print("\n9. Using ConfigBuilder")
    print("-" * 40)

    config = (ConfigBuilder()
        .with_gpu(GpuConfig.cuda())
        .with_llm(LlmConfig(model_path="./model.gguf"))
        .with_database(DatabaseConfig.in_memory())
        .with_observability(ObservabilityConfig(logging_level="info"))
        .build())

    print(f"   Builder config enabled: {config.enabled_features()}")

    # Full builder example
    full_config = (ConfigBuilder()
        .with_defaults()
        .build())
    print(f"   Full builder config: {full_config.enabled_features()}")

    # Combined Configuration
    print("\n10. Combined Configuration")
    print("-" * 40)

    combined = Config(
        gpu=GpuConfig(backend=GpuBackend.CUDA),
        ai=AiConfig(
            llm=LlmConfig(context_size=4096),
            embeddings=EmbeddingsConfig(dimension=384),
            agents=AgentsConfig(max_agents=8),
        ),
        database=DatabaseConfig(path="./vectors.db"),
        network=NetworkConfig(port=9000),
        observability=ObservabilityConfig(metrics_enabled=True),
    )

    print(f"   GPU backend: {combined.gpu.backend.name}")
    print(f"   LLM context: {combined.ai.llm.context_size}")
    print(f"   Embeddings dim: {combined.ai.embeddings.dimension}")
    print(f"   Max agents: {combined.ai.agents.max_agents}")
    print(f"   DB path: {combined.database.path}")
    print(f"   Network port: {combined.network.port}")

    # Feature checking
    print("\n11. Feature Checking")
    print("-" * 40)

    print(f"   GPU enabled: {combined.is_enabled('gpu')}")
    print(f"   AI enabled: {combined.is_enabled('ai')}")
    print(f"   LLM enabled: {combined.is_enabled('llm')}")
    print(f"   Database enabled: {combined.is_enabled('database')}")
    print(f"   Training enabled: {combined.is_enabled('training')}")

    # Serialization
    print("\n12. Configuration Serialization")
    print("-" * 40)

    import json

    config_dict = combined.to_dict()
    json_str = json.dumps(config_dict, indent=2)
    print(f"   JSON representation (truncated):")
    lines = json_str.split('\n')
    for line in lines[:10]:
        print(f"   {line}")
    if len(lines) > 10:
        print("   ...")

    # Initialize framework with configuration
    print("\n13. Initialize with Configuration")
    print("-" * 40)

    # Create a simple config for initialization
    init_config = Config(
        gpu=GpuConfig.cpu_only(),
        database=DatabaseConfig.in_memory(),
    )

    abi.init(init_config)
    print(f"   Framework initialized with custom config")
    print(f"   Version: {abi.version()}")

    # Cleanup
    abi.shutdown()
    print("   Framework shut down")

    print("\n" + "=" * 60)
    print("Configuration Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
