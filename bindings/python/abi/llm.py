"""
ABI Framework LLM Module

Provides local LLM inference capabilities supporting GGUF models
and transformer architectures.
"""

from typing import List, Optional, Dict, Any, Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
import os


class SamplingMethod(Enum):
    """Token sampling methods."""
    GREEDY = auto()
    TOP_K = auto()
    TOP_P = auto()
    TEMPERATURE = auto()
    MIROSTAT = auto()
    MIROSTAT_V2 = auto()


@dataclass
class InferenceConfig:
    """
    Configuration for LLM inference.

    Attributes:
        max_context_length: Maximum context length in tokens
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 = greedy, 1.0 = default)
        top_p: Top-p nucleus sampling threshold
        top_k: Top-k sampling (0 = disabled)
        repetition_penalty: Repetition penalty (1.0 = disabled)
        use_gpu: Use GPU acceleration if available
        num_threads: Number of CPU threads (0 = auto-detect)
        streaming: Enable streaming output
        batch_size: Batch size for prefill

    Example:
        >>> config = InferenceConfig(temperature=0.7, top_p=0.9)
        >>> config = InferenceConfig(max_new_tokens=512, use_gpu=True)
    """
    max_context_length: int = 2048
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.1
    use_gpu: bool = True
    num_threads: int = 0
    streaming: bool = True
    batch_size: int = 512

    @classmethod
    def default(cls) -> "InferenceConfig":
        """Create default inference configuration."""
        return cls()

    @classmethod
    def greedy(cls) -> "InferenceConfig":
        """Create greedy sampling configuration."""
        return cls(temperature=0.0, top_k=1)

    @classmethod
    def creative(cls) -> "InferenceConfig":
        """Create creative sampling configuration."""
        return cls(temperature=1.0, top_p=0.95, top_k=100)


@dataclass
class InferenceStats:
    """
    Statistics from LLM inference.

    Attributes:
        load_time_ns: Time to load model in nanoseconds
        prefill_time_ns: Time for prompt processing in nanoseconds
        decode_time_ns: Time for token generation in nanoseconds
        prompt_tokens: Number of prompt tokens processed
        generated_tokens: Number of tokens generated
        peak_memory_bytes: Peak memory usage in bytes
        used_gpu: Whether GPU was used
    """
    load_time_ns: int = 0
    prefill_time_ns: int = 0
    decode_time_ns: int = 0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    peak_memory_bytes: int = 0
    used_gpu: bool = False

    @property
    def prefill_tokens_per_second(self) -> float:
        """Calculate prefill throughput."""
        if self.prefill_time_ns == 0:
            return 0.0
        return self.prompt_tokens / (self.prefill_time_ns / 1_000_000_000)

    @property
    def decode_tokens_per_second(self) -> float:
        """Calculate decode throughput."""
        if self.decode_time_ns == 0:
            return 0.0
        return self.generated_tokens / (self.decode_time_ns / 1_000_000_000)

    @property
    def total_time_seconds(self) -> float:
        """Get total inference time in seconds."""
        return (self.prefill_time_ns + self.decode_time_ns) / 1_000_000_000

    def __repr__(self) -> str:
        return (
            f"InferenceStats(prefill={self.prefill_tokens_per_second:.1f} tok/s, "
            f"decode={self.decode_tokens_per_second:.1f} tok/s, "
            f"prompt={self.prompt_tokens}, generated={self.generated_tokens}, "
            f"gpu={self.used_gpu})"
        )


@dataclass
class ModelInfo:
    """
    Information about a loaded model.

    Attributes:
        name: Model name
        path: Model file path
        architecture: Model architecture (e.g., "llama", "mistral")
        parameters: Number of parameters
        context_length: Maximum context length
        vocab_size: Vocabulary size
        quantization: Quantization type (e.g., "Q4_0", "Q8_0")
        file_size_bytes: Model file size in bytes
    """
    name: str = ""
    path: str = ""
    architecture: str = ""
    parameters: int = 0
    context_length: int = 0
    vocab_size: int = 0
    quantization: str = ""
    file_size_bytes: int = 0


class LlmEngine:
    """
    High-level interface for loading and running LLM models.

    This class provides a pure Python implementation with optional
    native acceleration when the ABI library is available.

    Example:
        >>> engine = LlmEngine()
        >>> engine.load_model("./models/llama-7b.gguf")
        >>> response = engine.generate("Hello, how are you?")
        >>> print(response)

        >>> # Streaming generation
        >>> for token in engine.generate_streaming("Once upon a time"):
        ...     print(token, end="", flush=True)
    """

    def __init__(self, config: Optional[InferenceConfig] = None):
        """
        Initialize the LLM engine.

        Args:
            config: Inference configuration (uses defaults if not provided)
        """
        self._config = config or InferenceConfig.default()
        self._model: Optional[Dict[str, Any]] = None
        self._model_info: Optional[ModelInfo] = None
        self._stats = InferenceStats()
        self._lib = None

        # Try to load native library
        try:
            from . import _load_library
            self._lib = _load_library()
        except (ImportError, AttributeError):
            pass

    @property
    def config(self) -> InferenceConfig:
        """Get current inference configuration."""
        return self._config

    @config.setter
    def config(self, value: InferenceConfig) -> None:
        """Set inference configuration."""
        self._config = value

    @property
    def model_info(self) -> Optional[ModelInfo]:
        """Get information about the loaded model."""
        return self._model_info

    @property
    def stats(self) -> InferenceStats:
        """Get inference statistics."""
        return self._stats

    @property
    def is_loaded(self) -> bool:
        """Check if a model is loaded."""
        return self._model is not None

    def load_model(self, path: str) -> ModelInfo:
        """
        Load a model from a GGUF file.

        Args:
            path: Path to the GGUF model file

        Returns:
            ModelInfo object with model details

        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model format is invalid
        """
        import time

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        start_time = time.perf_counter_ns()

        # Get file size
        file_size = os.path.getsize(path)

        # Mock model loading - in production this would call native library
        self._model = {
            "path": path,
            "loaded": True,
        }

        self._model_info = ModelInfo(
            name=os.path.basename(path).replace(".gguf", ""),
            path=path,
            architecture="llama",  # Would be detected from GGUF
            parameters=0,  # Would be read from GGUF metadata
            context_length=self._config.max_context_length,
            vocab_size=32000,  # Would be read from GGUF metadata
            quantization="Q4_0",  # Would be read from GGUF metadata
            file_size_bytes=file_size,
        )

        self._stats.load_time_ns = time.perf_counter_ns() - start_time
        return self._model_info

    def unload_model(self) -> None:
        """Unload the current model and free resources."""
        self._model = None
        self._model_info = None
        self._stats = InferenceStats()

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate (overrides config)
            temperature: Sampling temperature (overrides config)
            top_p: Top-p sampling threshold (overrides config)
            top_k: Top-k sampling (overrides config)
            stop_sequences: List of sequences that stop generation

        Returns:
            Generated text

        Raises:
            RuntimeError: If no model is loaded
        """
        if not self.is_loaded:
            raise RuntimeError("No model loaded. Call load_model() first.")

        # Use provided values or fall back to config
        max_tokens = max_tokens or self._config.max_new_tokens
        temperature = temperature if temperature is not None else self._config.temperature
        top_p = top_p if top_p is not None else self._config.top_p
        top_k = top_k if top_k is not None else self._config.top_k

        import time
        start_time = time.perf_counter_ns()

        # Mock generation - in production this would call native inference
        # Placeholder response for development
        response = f"[Generated response for: {prompt[:50]}...]"

        self._stats.prefill_time_ns = time.perf_counter_ns() - start_time
        self._stats.prompt_tokens = len(prompt.split())
        self._stats.generated_tokens = len(response.split())

        return response

    def generate_streaming(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        callback: Optional[Callable[[str], None]] = None,
    ) -> Iterator[str]:
        """
        Generate text with streaming output.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            callback: Optional callback function called for each token

        Yields:
            Generated tokens one at a time

        Example:
            >>> for token in engine.generate_streaming("Hello"):
            ...     print(token, end="", flush=True)
        """
        if not self.is_loaded:
            raise RuntimeError("No model loaded. Call load_model() first.")

        max_tokens = max_tokens or self._config.max_new_tokens

        # Mock streaming generation
        tokens = ["[", "Generated", " ", "response", " ", "for", ":", " ", prompt[:20], "...]"]

        for token in tokens:
            if callback:
                callback(token)
            yield token

    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize text into token IDs.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        if not self.is_loaded:
            raise RuntimeError("No model loaded. Call load_model() first.")

        # Mock tokenization - returns approximate token count
        # In production, this would use the actual tokenizer
        words = text.split()
        return list(range(len(words)))

    def detokenize(self, tokens: List[int]) -> str:
        """
        Convert token IDs back to text.

        Args:
            tokens: List of token IDs

        Returns:
            Decoded text
        """
        if not self.is_loaded:
            raise RuntimeError("No model loaded. Call load_model() first.")

        # Mock detokenization
        return f"[Decoded {len(tokens)} tokens]"

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text without full tokenization.

        Args:
            text: Input text

        Returns:
            Approximate token count
        """
        # Rough approximation: ~4 characters per token
        return max(1, len(text) // 4)

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate a chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Assistant's response

        Example:
            >>> response = engine.chat([
            ...     {"role": "system", "content": "You are a helpful assistant."},
            ...     {"role": "user", "content": "Hello!"}
            ... ])
        """
        # Format messages into a prompt
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        prompt_parts.append("Assistant:")
        prompt = "\n".join(prompt_parts)

        return self.generate(prompt, max_tokens=max_tokens, temperature=temperature)


class LlmContext:
    """
    LLM context for framework integration.

    This class wraps the LLM engine to provide a consistent interface
    with other ABI framework modules.

    Example:
        >>> from abi.config import LlmConfig
        >>> ctx = LlmContext(LlmConfig(model_path="./model.gguf"))
        >>> response = ctx.generate("Hello!")
    """

    def __init__(self, config: Optional["LlmConfig"] = None):
        """
        Initialize the LLM context.

        Args:
            config: LLM configuration from abi.config
        """
        from .config import LlmConfig

        self._config = config or LlmConfig.defaults()
        self._engine: Optional[LlmEngine] = None

    @property
    def engine(self) -> LlmEngine:
        """Get or create the LLM engine."""
        if self._engine is None:
            inference_config = InferenceConfig(
                max_context_length=self._config.context_size,
                batch_size=self._config.batch_size,
                use_gpu=self._config.use_gpu,
                num_threads=self._config.threads or 0,
            )
            self._engine = LlmEngine(inference_config)

            # Auto-load model if path is provided
            if self._config.model_path:
                self._engine.load_model(self._config.model_path)

        return self._engine

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        return self.engine.generate(prompt, **kwargs)

    def tokenize(self, text: str) -> List[int]:
        """Tokenize text."""
        return self.engine.tokenize(text)

    def detokenize(self, tokens: List[int]) -> str:
        """Detokenize tokens."""
        return self.engine.detokenize(tokens)

    def close(self) -> None:
        """Close and release resources."""
        if self._engine:
            self._engine.unload_model()
            self._engine = None


# Convenience functions

def infer(model_path: str, prompt: str, **kwargs) -> str:
    """
    Quick inference helper.

    Args:
        model_path: Path to GGUF model file
        prompt: Input prompt
        **kwargs: Additional inference parameters

    Returns:
        Generated text

    Example:
        >>> result = infer("./model.gguf", "Hello, world!")
    """
    engine = LlmEngine()
    engine.load_model(model_path)
    return engine.generate(prompt, **kwargs)


def list_models(directory: str = "./models") -> List[str]:
    """
    List available GGUF models in a directory.

    Args:
        directory: Directory to search for models

    Returns:
        List of model file paths
    """
    models = []
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            if filename.endswith(".gguf"):
                models.append(os.path.join(directory, filename))
    return models


def is_enabled() -> bool:
    """Check if LLM features are available."""
    # In mock mode, always return True
    # In production, this would check native library availability
    return True
