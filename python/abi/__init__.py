"""
ABI Framework Python Bindings

High-performance AI/ML framework with GPU acceleration.
"""

import os
import sys
from typing import Dict, List, Optional, Union, Any
import numpy as np

# Import the compiled Zig extension
try:
    from . import _abi_core
except ImportError:
    # Fallback for development
    import warnings

    warnings.warn("ABI core extension not found. Using mock implementation.")
    _abi_core = None


class Framework:
    """
    Main ABI Framework interface.

    Provides access to all AI/ML capabilities including transformers,
    vector databases, reinforcement learning, and GPU acceleration.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ABI framework.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self._initialized = False

        if _abi_core:
            # Initialize with real Zig backend
            self._handle = _abi_core.abi_init(self.config)
            self._initialized = True
        else:
            # Mock implementation for development
            self._handle = None
            print("ABI Framework initialized (mock mode)")

    def __del__(self):
        """Clean up framework resources"""
        if hasattr(self, "_handle") and self._handle:
            if _abi_core:
                _abi_core.abi_deinit(self._handle)


class Transformer:
    """
    Transformer model for natural language processing.

    Supports encoding text into vector embeddings and various
    transformer-based tasks.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Create a transformer model.

        Args:
            config: Model configuration including:
                - vocab_size: Size of vocabulary
                - d_model: Model dimension
                - n_heads: Number of attention heads
                - n_layers: Number of transformer layers
                - max_seq_len: Maximum sequence length
        """
        self.config = config
        self._model = None

        if _abi_core:
            # Create real transformer model
            self._model = _abi_core.transformer_create(config)
        else:
            # Mock implementation
            print(f"Transformer created with config: {config}")

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text(s) into vector embeddings.

        Args:
            texts: Single text string or list of texts

        Returns:
            numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        if _abi_core and self._model:
            # Real encoding
            embeddings = _abi_core.transformer_encode(self._model, texts)
            return np.array(embeddings)
        else:
            # Mock embeddings
            batch_size = len(texts)
            embedding_dim = self.config.get("d_model", 512)
            return np.random.randn(batch_size, embedding_dim).astype(np.float32)


class VectorDatabase:
    """
    High-performance vector database for similarity search.

    Supports HNSW indexing, multiple distance metrics, and
    efficient nearest neighbor search.
    """

    def __init__(self, dimensions: int, distance_metric: str = "cosine"):
        """
        Create a vector database.

        Args:
            dimensions: Vector dimensionality
            distance_metric: Distance metric ('cosine', 'euclidean', 'dot')
        """
        self.dimensions = dimensions
        self.distance_metric = distance_metric
        self._db = None

        if _abi_core:
            # Create real database
            self._db = _abi_core.vector_db_create(dimensions, distance_metric)
        else:
            # Mock database
            print(f"Vector database created: {dimensions}D, metric={distance_metric}")

    def add(self, vectors: np.ndarray, ids: Optional[List[str]] = None) -> List[str]:
        """
        Add vectors to the database.

        Args:
            vectors: numpy array of shape (n_vectors, dimensions)
            ids: Optional list of string IDs

        Returns:
            List of assigned IDs
        """
        if ids is None:
            ids = [f"vec_{i}" for i in range(len(vectors))]

        if _abi_core and self._db:
            # Real database insertion
            return _abi_core.vector_db_add(self._db, vectors.tolist(), ids)
        else:
            # Mock insertion
            return ids

    def search(self, query: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.

        Args:
            query: Query vector
            top_k: Number of results to return

        Returns:
            List of results with 'id', 'score', and 'vector' keys
        """
        if _abi_core and self._db:
            # Real search
            results = _abi_core.vector_db_search(self._db, query.tolist(), top_k)
            return results
        else:
            # Mock search results
            results = []
            for i in range(min(top_k, 10)):
                results.append(
                    {
                        "id": f"mock_{i}",
                        "score": 1.0 - (i * 0.1),  # Decreasing scores
                        "vector": query.tolist(),
                    }
                )
            return results


class ReinforcementLearning:
    """
    Reinforcement learning algorithms and environments.

    Supports Q-learning, SARSA, and policy gradient methods.
    """

    def __init__(self, algorithm: str = "q_learning", **kwargs):
        """
        Create an RL agent.

        Args:
            algorithm: RL algorithm ('q_learning', 'sarsa', 'policy_gradient')
            **kwargs: Algorithm-specific parameters
        """
        self.algorithm = algorithm
        self.config = kwargs
        self._agent = None

        if _abi_core:
            # Create real RL agent
            self._agent = _abi_core.rl_create_agent(algorithm, kwargs)
        else:
            # Mock agent
            print(f"RL agent created: {algorithm} with config {kwargs}")

    def choose_action(self, state: Union[List[float], np.ndarray]) -> int:
        """
        Choose an action given the current state.

        Args:
            state: Current environment state

        Returns:
            Action index
        """
        state_list = state.tolist() if hasattr(state, "tolist") else state

        if _abi_core and self._agent:
            return _abi_core.rl_choose_action(self._agent, state_list)
        else:
            # Mock action selection
            return 0  # Always choose first action in mock mode

    def learn(
        self,
        state: Union[List[float], np.ndarray],
        action: int,
        reward: float,
        next_state: Union[List[float], np.ndarray],
    ):
        """
        Update the agent with experience.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        state_list = state.tolist() if hasattr(state, "tolist") else state
        next_state_list = (
            next_state.tolist() if hasattr(next_state, "tolist") else next_state
        )

        if _abi_core and self._agent:
            _abi_core.rl_learn(self._agent, state_list, action, reward, next_state_list)
        # Mock learning - no-op


# Convenience functions
def create_framework(config: Optional[Dict[str, Any]] = None) -> Framework:
    """Create and initialize the ABI framework."""
    return Framework(config)


def create_transformer(config: Dict[str, Any]) -> Transformer:
    """Create a transformer model."""
    return Transformer(config)


def create_vector_db(
    dimensions: int, distance_metric: str = "cosine"
) -> VectorDatabase:
    """Create a vector database."""
    return VectorDatabase(dimensions, distance_metric)


def create_rl_agent(algorithm: str = "q_learning", **kwargs) -> ReinforcementLearning:
    """Create a reinforcement learning agent."""
    return ReinforcementLearning(algorithm, **kwargs)


# Version information
__version__ = "0.2.0"
__author__ = "ABI Framework Team"
