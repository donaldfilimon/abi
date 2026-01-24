"""
ABI Framework LLM Streaming Bindings

Low-level ctypes bindings for streaming inference and training FFI.
"""

import ctypes
from typing import Optional, Callable
from dataclasses import dataclass


# =============================================================================
# C-Compatible Structures
# =============================================================================

class CTokenEvent(ctypes.Structure):
    """C-compatible token event structure."""
    _fields_ = [
        ("token_id", ctypes.c_uint32),
        ("text_ptr", ctypes.c_char_p),
        ("text_len", ctypes.c_uint32),
        ("position", ctypes.c_uint32),
        ("is_final", ctypes.c_bool),
        ("timestamp_ns", ctypes.c_uint64),
    ]


class CStreamingConfig(ctypes.Structure):
    """C-compatible streaming config structure."""
    _fields_ = [
        ("max_tokens", ctypes.c_uint32),
        ("temperature", ctypes.c_float),
        ("top_k", ctypes.c_uint32),
        ("top_p", ctypes.c_float),
        ("repetition_penalty", ctypes.c_float),
        ("seed", ctypes.c_uint64),
    ]


class CTrainingConfig(ctypes.Structure):
    """C-compatible training config structure."""
    _fields_ = [
        ("epochs", ctypes.c_uint32),
        ("batch_size", ctypes.c_uint32),
        ("learning_rate", ctypes.c_float),
        ("optimizer", ctypes.c_uint32),  # 0=sgd, 1=adam, 2=adamw
        ("weight_decay", ctypes.c_float),
        ("gradient_clip_norm", ctypes.c_float),
        ("warmup_steps", ctypes.c_uint32),
        ("checkpoint_interval", ctypes.c_uint32),
    ]


class CTrainingMetrics(ctypes.Structure):
    """C-compatible training metrics structure (per step)."""
    _fields_ = [
        ("step", ctypes.c_uint32),
        ("epoch", ctypes.c_uint32),
        ("loss", ctypes.c_float),
        ("accuracy", ctypes.c_float),
        ("learning_rate", ctypes.c_float),
        ("gradient_norm", ctypes.c_float),
    ]


class CTrainingReport(ctypes.Structure):
    """C-compatible training report structure (final)."""
    _fields_ = [
        ("epochs", ctypes.c_uint32),
        ("batches", ctypes.c_uint32),
        ("final_loss", ctypes.c_float),
        ("final_accuracy", ctypes.c_float),
        ("best_loss", ctypes.c_float),
        ("gradient_updates", ctypes.c_uint64),
        ("checkpoints_saved", ctypes.c_uint32),
        ("early_stopped", ctypes.c_bool),
        ("total_time_ms", ctypes.c_uint64),
    ]


# =============================================================================
# Python Data Classes
# =============================================================================

@dataclass
class TokenEvent:
    """Token event from streaming inference."""
    token_id: int
    text: Optional[str]
    position: int
    is_final: bool
    timestamp_ns: int

    @classmethod
    def from_c(cls, c_event: CTokenEvent) -> "TokenEvent":
        """Create from C struct."""
        text = None
        if c_event.text_ptr and c_event.text_len > 0:
            try:
                text = c_event.text_ptr[:c_event.text_len].decode("utf-8", errors="replace")
            except (ValueError, AttributeError):
                pass
        return cls(
            token_id=c_event.token_id,
            text=text,
            position=c_event.position,
            is_final=c_event.is_final,
            timestamp_ns=c_event.timestamp_ns,
        )


@dataclass
class StreamingConfig:
    """Configuration for streaming inference."""
    max_tokens: int = 256
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    seed: int = 0

    def to_c(self) -> CStreamingConfig:
        """Convert to C struct."""
        return CStreamingConfig(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            seed=self.seed,
        )


@dataclass
class TrainingMetrics:
    """Per-step training metrics."""
    step: int
    epoch: int
    loss: float
    accuracy: float
    learning_rate: float
    gradient_norm: float

    @classmethod
    def from_c(cls, c_metrics: CTrainingMetrics) -> "TrainingMetrics":
        """Create from C struct."""
        return cls(
            step=c_metrics.step,
            epoch=c_metrics.epoch,
            loss=c_metrics.loss,
            accuracy=c_metrics.accuracy,
            learning_rate=c_metrics.learning_rate,
            gradient_norm=c_metrics.gradient_norm,
        )


@dataclass
class TrainingReport:
    """Final training report."""
    epochs: int
    batches: int
    final_loss: float
    final_accuracy: float
    best_loss: float
    gradient_updates: int
    checkpoints_saved: int
    early_stopped: bool
    total_time_ms: int

    @classmethod
    def from_c(cls, c_report: CTrainingReport) -> "TrainingReport":
        """Create from C struct."""
        return cls(
            epochs=c_report.epochs,
            batches=c_report.batches,
            final_loss=c_report.final_loss,
            final_accuracy=c_report.final_accuracy,
            best_loss=c_report.best_loss,
            gradient_updates=c_report.gradient_updates,
            checkpoints_saved=c_report.checkpoints_saved,
            early_stopped=c_report.early_stopped,
            total_time_ms=c_report.total_time_ms,
        )

    @property
    def total_time_seconds(self) -> float:
        """Get total time in seconds."""
        return self.total_time_ms / 1000.0


# =============================================================================
# Library Function Setup
# =============================================================================

def _setup_streaming_functions(lib) -> None:
    """Configure streaming FFI function signatures."""
    if lib is None:
        return

    # abi_llm_stream_create
    if hasattr(lib, "abi_llm_stream_create"):
        lib.abi_llm_stream_create.restype = ctypes.c_int32
        lib.abi_llm_stream_create.argtypes = [
            ctypes.c_char_p,  # prompt
            ctypes.c_uint32,  # max_tokens
            ctypes.c_float,   # temperature
        ]

    # abi_llm_stream_create_ex
    if hasattr(lib, "abi_llm_stream_create_ex"):
        lib.abi_llm_stream_create_ex.restype = ctypes.c_int32
        lib.abi_llm_stream_create_ex.argtypes = [
            ctypes.c_char_p,                    # prompt
            ctypes.POINTER(CStreamingConfig),   # config
        ]

    # abi_llm_stream_next
    if hasattr(lib, "abi_llm_stream_next"):
        lib.abi_llm_stream_next.restype = ctypes.c_bool
        lib.abi_llm_stream_next.argtypes = [
            ctypes.c_int32,                 # stream_id
            ctypes.POINTER(CTokenEvent),    # event_out
        ]

    # abi_llm_stream_cancel
    if hasattr(lib, "abi_llm_stream_cancel"):
        lib.abi_llm_stream_cancel.restype = None
        lib.abi_llm_stream_cancel.argtypes = [ctypes.c_int32]

    # abi_llm_stream_destroy
    if hasattr(lib, "abi_llm_stream_destroy"):
        lib.abi_llm_stream_destroy.restype = None
        lib.abi_llm_stream_destroy.argtypes = [ctypes.c_int32]

    # abi_train_create
    if hasattr(lib, "abi_train_create"):
        lib.abi_train_create.restype = ctypes.c_int32
        lib.abi_train_create.argtypes = [ctypes.POINTER(CTrainingConfig)]

    # abi_train_step
    if hasattr(lib, "abi_train_step"):
        lib.abi_train_step.restype = ctypes.c_bool
        lib.abi_train_step.argtypes = [
            ctypes.c_int32,                   # trainer_id
            ctypes.POINTER(CTrainingMetrics), # metrics_out
        ]

    # abi_train_save_checkpoint
    if hasattr(lib, "abi_train_save_checkpoint"):
        lib.abi_train_save_checkpoint.restype = ctypes.c_int32
        lib.abi_train_save_checkpoint.argtypes = [
            ctypes.c_int32,     # trainer_id
            ctypes.c_char_p,    # path
        ]

    # abi_train_get_report
    if hasattr(lib, "abi_train_get_report"):
        lib.abi_train_get_report.restype = None
        lib.abi_train_get_report.argtypes = [
            ctypes.c_int32,                   # trainer_id
            ctypes.POINTER(CTrainingReport),  # report_out
        ]

    # abi_train_destroy
    if hasattr(lib, "abi_train_destroy"):
        lib.abi_train_destroy.restype = None
        lib.abi_train_destroy.argtypes = [ctypes.c_int32]

    # abi_get_last_error
    if hasattr(lib, "abi_get_last_error"):
        lib.abi_get_last_error.restype = ctypes.c_char_p
        lib.abi_get_last_error.argtypes = []


# =============================================================================
# Low-Level Wrapper Functions
# =============================================================================

def _get_last_error(lib) -> str:
    """Get last error message from library."""
    if lib is None or not hasattr(lib, "abi_get_last_error"):
        return "Library not loaded"
    result = lib.abi_get_last_error()
    if result:
        return result.decode("utf-8", errors="replace")
    return "Unknown error"


def _stream_create(lib, prompt: str, max_tokens: int, temperature: float) -> int:
    """Create a streaming session."""
    if lib is None:
        raise RuntimeError("Native library not loaded")

    prompt_bytes = prompt.encode("utf-8")
    handle_id = lib.abi_llm_stream_create(prompt_bytes, max_tokens, temperature)

    if handle_id < 0:
        raise RuntimeError(f"Failed to create stream: {_get_last_error(lib)}")

    return handle_id


def _stream_create_ex(lib, prompt: str, config: StreamingConfig) -> int:
    """Create a streaming session with full config."""
    if lib is None:
        raise RuntimeError("Native library not loaded")

    prompt_bytes = prompt.encode("utf-8")
    c_config = config.to_c()
    handle_id = lib.abi_llm_stream_create_ex(prompt_bytes, ctypes.byref(c_config))

    if handle_id < 0:
        raise RuntimeError(f"Failed to create stream: {_get_last_error(lib)}")

    return handle_id


def _stream_next(lib, stream_id: int) -> Optional[TokenEvent]:
    """Get next token from stream."""
    if lib is None:
        return None

    c_event = CTokenEvent()
    has_more = lib.abi_llm_stream_next(stream_id, ctypes.byref(c_event))

    if not has_more and c_event.is_final:
        return TokenEvent.from_c(c_event)  # Return final event
    elif has_more:
        return TokenEvent.from_c(c_event)

    return None


def _stream_cancel(lib, stream_id: int) -> None:
    """Cancel an active stream."""
    if lib is not None and hasattr(lib, "abi_llm_stream_cancel"):
        lib.abi_llm_stream_cancel(stream_id)


def _stream_destroy(lib, stream_id: int) -> None:
    """Destroy a stream and free resources."""
    if lib is not None and hasattr(lib, "abi_llm_stream_destroy"):
        lib.abi_llm_stream_destroy(stream_id)


def _train_create(lib, config: "TrainingConfig") -> int:
    """Create a training session."""
    if lib is None:
        raise RuntimeError("Native library not loaded")

    c_config = CTrainingConfig(
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        optimizer={"sgd": 0, "adam": 1, "adamw": 2}.get(config.optimizer, 2),
        weight_decay=config.weight_decay,
        gradient_clip_norm=config.gradient_clip_norm,
        warmup_steps=config.warmup_steps,
        checkpoint_interval=config.checkpoint_interval,
    )

    handle_id = lib.abi_train_create(ctypes.byref(c_config))

    if handle_id < 0:
        raise RuntimeError(f"Failed to create trainer: {_get_last_error(lib)}")

    return handle_id


def _train_step(lib, trainer_id: int) -> Optional[TrainingMetrics]:
    """Run one training step."""
    if lib is None:
        return None

    c_metrics = CTrainingMetrics()
    has_more = lib.abi_train_step(trainer_id, ctypes.byref(c_metrics))

    if has_more or c_metrics.step > 0:
        return TrainingMetrics.from_c(c_metrics)

    return None


def _train_save_checkpoint(lib, trainer_id: int, path: str) -> None:
    """Save training checkpoint."""
    if lib is None:
        raise RuntimeError("Native library not loaded")

    path_bytes = path.encode("utf-8")
    result = lib.abi_train_save_checkpoint(trainer_id, path_bytes)

    if result < 0:
        raise RuntimeError(f"Failed to save checkpoint: {_get_last_error(lib)}")


def _train_get_report(lib, trainer_id: int) -> TrainingReport:
    """Get final training report."""
    if lib is None:
        raise RuntimeError("Native library not loaded")

    c_report = CTrainingReport()
    lib.abi_train_get_report(trainer_id, ctypes.byref(c_report))
    return TrainingReport.from_c(c_report)


def _train_destroy(lib, trainer_id: int) -> None:
    """Destroy training session."""
    if lib is not None and hasattr(lib, "abi_train_destroy"):
        lib.abi_train_destroy(trainer_id)
