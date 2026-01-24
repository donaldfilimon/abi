"""
ABI Framework Configuration Module

Provides configuration classes for the ABI framework including
GPU, AI, LLM, database, and network settings.
"""

from typing import Optional, Dict, Any, List
from enum import Enum, auto
from dataclasses import dataclass, field


class GpuBackend(Enum):
    """Available GPU backends."""
    AUTO = auto()
    VULKAN = auto()
    CUDA = auto()
    METAL = auto()
    WEBGPU = auto()
    OPENGL = auto()
    FPGA = auto()
    CPU = auto()


class IndexType(Enum):
    """Vector database index types."""
    HNSW = auto()
    IVF_PQ = auto()
    FLAT = auto()


class DistanceMetric(Enum):
    """Distance metrics for vector similarity."""
    COSINE = auto()
    EUCLIDEAN = auto()
    DOT_PRODUCT = auto()


class Optimizer(Enum):
    """Training optimizers."""
    SGD = auto()
    ADAM = auto()
    ADAMW = auto()
    RMSPROP = auto()


@dataclass
class GpuRecoveryConfig:
    """GPU failure recovery configuration."""
    enabled: bool = True
    max_retries: int = 3
    fallback_to_cpu: bool = True


@dataclass
class GpuConfig:
    """
    GPU acceleration configuration.

    Attributes:
        backend: GPU backend to use (auto-detect by default)
        device_index: Preferred device index (0 = first available)
        memory_limit: Maximum GPU memory to use in bytes (None = no limit)
        async_enabled: Enable asynchronous operations
        cache_kernels: Enable kernel caching for performance
        recovery: Recovery settings for GPU failures

    Example:
        >>> config = GpuConfig(backend=GpuBackend.CUDA, device_index=0)
        >>> config = GpuConfig(memory_limit=4 * 1024 * 1024 * 1024)  # 4GB
    """
    backend: GpuBackend = GpuBackend.AUTO
    device_index: int = 0
    memory_limit: Optional[int] = None
    async_enabled: bool = True
    cache_kernels: bool = True
    recovery: GpuRecoveryConfig = field(default_factory=GpuRecoveryConfig)

    @classmethod
    def defaults(cls) -> "GpuConfig":
        """Create default GPU configuration."""
        return cls()

    @classmethod
    def cuda(cls, device_index: int = 0) -> "GpuConfig":
        """Create CUDA-specific configuration."""
        return cls(backend=GpuBackend.CUDA, device_index=device_index)

    @classmethod
    def vulkan(cls, device_index: int = 0) -> "GpuConfig":
        """Create Vulkan-specific configuration."""
        return cls(backend=GpuBackend.VULKAN, device_index=device_index)

    @classmethod
    def metal(cls) -> "GpuConfig":
        """Create Metal-specific configuration (macOS)."""
        return cls(backend=GpuBackend.METAL)

    @classmethod
    def cpu_only(cls) -> "GpuConfig":
        """Create CPU-only configuration (disable GPU)."""
        return cls(backend=GpuBackend.CPU)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "backend": self.backend.name.lower(),
            "device_index": self.device_index,
            "memory_limit": self.memory_limit,
            "async_enabled": self.async_enabled,
            "cache_kernels": self.cache_kernels,
            "recovery": {
                "enabled": self.recovery.enabled,
                "max_retries": self.recovery.max_retries,
                "fallback_to_cpu": self.recovery.fallback_to_cpu,
            },
        }


@dataclass
class LlmConfig:
    """
    LLM inference configuration.

    Attributes:
        model_path: Path to model file (GGUF format)
        model_name: Model name from registry
        context_size: Context window size in tokens
        threads: Number of CPU threads (None = auto-detect)
        use_gpu: Use GPU acceleration if available
        batch_size: Batch size for inference

    Example:
        >>> config = LlmConfig(model_path="./models/llama-7b.gguf")
        >>> config = LlmConfig(context_size=4096, use_gpu=True)
    """
    model_path: Optional[str] = None
    model_name: str = "gpt2"
    context_size: int = 2048
    threads: Optional[int] = None
    use_gpu: bool = True
    batch_size: int = 512

    @classmethod
    def defaults(cls) -> "LlmConfig":
        """Create default LLM configuration."""
        return cls()

    @classmethod
    def from_model(cls, model_path: str, **kwargs) -> "LlmConfig":
        """Create configuration from a model file path."""
        return cls(model_path=model_path, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_path": self.model_path,
            "model_name": self.model_name,
            "context_size": self.context_size,
            "threads": self.threads,
            "use_gpu": self.use_gpu,
            "batch_size": self.batch_size,
        }


@dataclass
class EmbeddingsConfig:
    """
    Embeddings generation configuration.

    Attributes:
        model: Embedding model to use
        dimension: Output embedding dimension
        normalize: Normalize output vectors
    """
    model: str = "default"
    dimension: int = 384
    normalize: bool = True

    @classmethod
    def defaults(cls) -> "EmbeddingsConfig":
        """Create default embeddings configuration."""
        return cls()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model": self.model,
            "dimension": self.dimension,
            "normalize": self.normalize,
        }


@dataclass
class AgentsConfig:
    """
    Agent runtime configuration.

    Attributes:
        max_agents: Maximum concurrent agents
        timeout_ms: Default agent timeout in milliseconds
        persistent_memory: Enable agent memory persistence
    """
    max_agents: int = 16
    timeout_ms: int = 30000
    persistent_memory: bool = False

    @classmethod
    def defaults(cls) -> "AgentsConfig":
        """Create default agents configuration."""
        return cls()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "max_agents": self.max_agents,
            "timeout_ms": self.timeout_ms,
            "persistent_memory": self.persistent_memory,
        }


@dataclass
class TrainingConfig:
    """
    Training pipeline configuration.

    Attributes:
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        optimizer: Optimizer to use
        checkpoint_dir: Directory for checkpoints
        checkpoint_frequency: Checkpoint frequency in epochs
    """
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: Optimizer = Optimizer.ADAMW
    checkpoint_dir: Optional[str] = None
    checkpoint_frequency: int = 1

    @classmethod
    def defaults(cls) -> "TrainingConfig":
        """Create default training configuration."""
        return cls()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer.name.lower(),
            "checkpoint_dir": self.checkpoint_dir,
            "checkpoint_frequency": self.checkpoint_frequency,
        }


@dataclass
class AiConfig:
    """
    AI configuration with independent sub-features.

    Attributes:
        llm: LLM inference settings
        embeddings: Embeddings generation settings
        agents: Agent runtime settings
        training: Training pipeline settings

    Example:
        >>> config = AiConfig(llm=LlmConfig(model_path="./model.gguf"))
        >>> config = AiConfig.llm_only(LlmConfig(context_size=4096))
    """
    llm: Optional[LlmConfig] = None
    embeddings: Optional[EmbeddingsConfig] = None
    agents: Optional[AgentsConfig] = None
    training: Optional[TrainingConfig] = None

    @classmethod
    def defaults(cls) -> "AiConfig":
        """Create default AI configuration with all sub-features."""
        return cls(
            llm=LlmConfig.defaults(),
            embeddings=EmbeddingsConfig.defaults(),
            agents=AgentsConfig.defaults(),
        )

    @classmethod
    def llm_only(cls, llm_config: Optional[LlmConfig] = None) -> "AiConfig":
        """Create configuration with only LLM enabled."""
        return cls(llm=llm_config or LlmConfig.defaults())

    @classmethod
    def embeddings_only(cls, config: Optional[EmbeddingsConfig] = None) -> "AiConfig":
        """Create configuration with only embeddings enabled."""
        return cls(embeddings=config or EmbeddingsConfig.defaults())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "llm": self.llm.to_dict() if self.llm else None,
            "embeddings": self.embeddings.to_dict() if self.embeddings else None,
            "agents": self.agents.to_dict() if self.agents else None,
            "training": self.training.to_dict() if self.training else None,
        }


@dataclass
class DatabaseConfig:
    """
    Vector database configuration.

    Attributes:
        path: Database file path
        index_type: Index type for vector search
        wal_enabled: Enable write-ahead logging
        cache_size: Cache size in bytes
        auto_optimize: Auto-optimize on startup
        dimensions: Vector dimensions (0 = auto-detect)
        distance_metric: Distance metric for similarity

    Example:
        >>> config = DatabaseConfig(path="./vectors.db", dimensions=384)
        >>> config = DatabaseConfig.in_memory()
    """
    path: str = "./abi.db"
    index_type: IndexType = IndexType.HNSW
    wal_enabled: bool = True
    cache_size: int = 64 * 1024 * 1024  # 64MB
    auto_optimize: bool = False
    dimensions: int = 0
    distance_metric: DistanceMetric = DistanceMetric.COSINE

    @classmethod
    def defaults(cls) -> "DatabaseConfig":
        """Create default database configuration."""
        return cls()

    @classmethod
    def in_memory(cls) -> "DatabaseConfig":
        """Create in-memory database configuration."""
        return cls(path=":memory:", wal_enabled=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "path": self.path,
            "index_type": self.index_type.name.lower(),
            "wal_enabled": self.wal_enabled,
            "cache_size": self.cache_size,
            "auto_optimize": self.auto_optimize,
            "dimensions": self.dimensions,
            "distance_metric": self.distance_metric.name.lower(),
        }


@dataclass
class NetworkConfig:
    """
    Network/distributed compute configuration.

    Attributes:
        host: Network host address
        port: Network port
        cluster_enabled: Enable cluster mode
        max_connections: Maximum connections
    """
    host: str = "127.0.0.1"
    port: int = 8080
    cluster_enabled: bool = False
    max_connections: int = 100

    @classmethod
    def defaults(cls) -> "NetworkConfig":
        """Create default network configuration."""
        return cls()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "host": self.host,
            "port": self.port,
            "cluster_enabled": self.cluster_enabled,
            "max_connections": self.max_connections,
        }


@dataclass
class ObservabilityConfig:
    """
    Observability and monitoring configuration.

    Attributes:
        metrics_enabled: Enable metrics collection
        tracing_enabled: Enable distributed tracing
        logging_level: Logging level (debug, info, warn, error)
    """
    metrics_enabled: bool = True
    tracing_enabled: bool = False
    logging_level: str = "info"

    @classmethod
    def defaults(cls) -> "ObservabilityConfig":
        """Create default observability configuration."""
        return cls()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metrics_enabled": self.metrics_enabled,
            "tracing_enabled": self.tracing_enabled,
            "logging_level": self.logging_level,
        }


@dataclass
class Config:
    """
    Unified configuration for the ABI framework.

    Each field being non-None enables that feature with the specified settings.
    A None field means the feature is disabled.

    Example:
        >>> # Minimal configuration
        >>> config = Config()

        >>> # Full configuration with all features
        >>> config = Config.defaults()

        >>> # Custom configuration
        >>> config = Config(
        ...     gpu=GpuConfig.cuda(),
        ...     ai=AiConfig.llm_only(LlmConfig(model_path="./model.gguf")),
        ...     database=DatabaseConfig(path="./vectors.db"),
        ... )
    """
    gpu: Optional[GpuConfig] = None
    ai: Optional[AiConfig] = None
    database: Optional[DatabaseConfig] = None
    network: Optional[NetworkConfig] = None
    observability: Optional[ObservabilityConfig] = None

    @classmethod
    def defaults(cls) -> "Config":
        """Create configuration with all features enabled with defaults."""
        return cls(
            gpu=GpuConfig.defaults(),
            ai=AiConfig.defaults(),
            database=DatabaseConfig.defaults(),
            network=NetworkConfig.defaults(),
            observability=ObservabilityConfig.defaults(),
        )

    @classmethod
    def minimal(cls) -> "Config":
        """Create minimal configuration with no features enabled."""
        return cls()

    def is_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled."""
        feature_map = {
            "gpu": self.gpu,
            "ai": self.ai,
            "llm": self.ai.llm if self.ai else None,
            "embeddings": self.ai.embeddings if self.ai else None,
            "agents": self.ai.agents if self.ai else None,
            "training": self.ai.training if self.ai else None,
            "database": self.database,
            "network": self.network,
            "observability": self.observability,
        }
        return feature_map.get(feature) is not None

    def enabled_features(self) -> List[str]:
        """Get list of enabled features."""
        features = []
        if self.gpu:
            features.append("gpu")
        if self.ai:
            features.append("ai")
            if self.ai.llm:
                features.append("llm")
            if self.ai.embeddings:
                features.append("embeddings")
            if self.ai.agents:
                features.append("agents")
            if self.ai.training:
                features.append("training")
        if self.database:
            features.append("database")
        if self.network:
            features.append("network")
        if self.observability:
            features.append("observability")
        return features

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "gpu": self.gpu.to_dict() if self.gpu else None,
            "ai": self.ai.to_dict() if self.ai else None,
            "database": self.database.to_dict() if self.database else None,
            "network": self.network.to_dict() if self.network else None,
            "observability": self.observability.to_dict() if self.observability else None,
        }


class ConfigBuilder:
    """
    Fluent builder for constructing Config instances.

    Example:
        >>> config = (ConfigBuilder()
        ...     .with_gpu(GpuConfig.cuda())
        ...     .with_ai(AiConfig.llm_only())
        ...     .with_database(DatabaseConfig.defaults())
        ...     .build())
    """

    def __init__(self):
        self._config = Config()

    def with_defaults(self) -> "ConfigBuilder":
        """Start with default configuration for all features."""
        self._config = Config.defaults()
        return self

    def with_gpu(self, config: Optional[GpuConfig] = None) -> "ConfigBuilder":
        """Enable GPU with specified or default configuration."""
        self._config.gpu = config or GpuConfig.defaults()
        return self

    def with_ai(self, config: Optional[AiConfig] = None) -> "ConfigBuilder":
        """Enable AI with specified or default configuration."""
        self._config.ai = config or AiConfig.defaults()
        return self

    def with_llm(self, config: Optional[LlmConfig] = None) -> "ConfigBuilder":
        """Enable LLM (creates AI config if needed)."""
        if self._config.ai is None:
            self._config.ai = AiConfig()
        self._config.ai.llm = config or LlmConfig.defaults()
        return self

    def with_database(self, config: Optional[DatabaseConfig] = None) -> "ConfigBuilder":
        """Enable database with specified or default configuration."""
        self._config.database = config or DatabaseConfig.defaults()
        return self

    def with_network(self, config: Optional[NetworkConfig] = None) -> "ConfigBuilder":
        """Enable network with specified or default configuration."""
        self._config.network = config or NetworkConfig.defaults()
        return self

    def with_observability(self, config: Optional[ObservabilityConfig] = None) -> "ConfigBuilder":
        """Enable observability with specified or default configuration."""
        self._config.observability = config or ObservabilityConfig.defaults()
        return self

    def build(self) -> Config:
        """Build and return the configuration."""
        return self._config
