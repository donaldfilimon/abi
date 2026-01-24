"""
ABI Framework Database Module

Provides vector database capabilities with HNSW indexing,
hybrid search, and batch operations.
"""

from typing import List, Optional, Dict, Any, Union, Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
import json
import os
import math


class DistanceMetric(Enum):
    """Distance metrics for vector similarity."""
    COSINE = auto()
    EUCLIDEAN = auto()
    DOT_PRODUCT = auto()


class IndexType(Enum):
    """Vector index types."""
    HNSW = auto()
    IVF_PQ = auto()
    FLAT = auto()


@dataclass
class DatabaseConfig:
    """
    Configuration for vector database.

    Attributes:
        path: Database file path
        dimensions: Vector dimensions (0 = auto-detect from first vector)
        distance_metric: Distance metric for similarity
        index_type: Index type for vector search
        ef_construction: HNSW construction parameter
        max_connections: HNSW max connections per node
        cache_size: Cache size in bytes
        wal_enabled: Enable write-ahead logging

    Example:
        >>> config = DatabaseConfig(dimensions=384, distance_metric=DistanceMetric.COSINE)
    """
    path: str = "./vectors.db"
    dimensions: int = 0
    distance_metric: DistanceMetric = DistanceMetric.COSINE
    index_type: IndexType = IndexType.HNSW
    ef_construction: int = 200
    max_connections: int = 16
    cache_size: int = 64 * 1024 * 1024  # 64MB
    wal_enabled: bool = True

    @classmethod
    def defaults(cls) -> "DatabaseConfig":
        """Create default configuration."""
        return cls()

    @classmethod
    def in_memory(cls, dimensions: int = 0) -> "DatabaseConfig":
        """Create in-memory configuration."""
        return cls(path=":memory:", dimensions=dimensions, wal_enabled=False)


@dataclass
class SearchResult:
    """
    Result from a vector search.

    Attributes:
        id: Vector ID
        score: Similarity score
        vector: Vector data (if include_vectors=True)
        metadata: Associated metadata
    """
    id: int
    score: float
    vector: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __repr__(self) -> str:
        return f"SearchResult(id={self.id}, score={self.score:.4f})"


@dataclass
class DatabaseStats:
    """
    Database statistics.

    Attributes:
        vector_count: Number of vectors stored
        dimensions: Vector dimensions
        index_type: Index type used
        memory_usage_bytes: Approximate memory usage
        index_build_time_ms: Time to build index
    """
    vector_count: int = 0
    dimensions: int = 0
    index_type: str = "hnsw"
    memory_usage_bytes: int = 0
    index_build_time_ms: float = 0.0


@dataclass
class BatchResult:
    """
    Result from a batch operation.

    Attributes:
        success_count: Number of successful operations
        error_count: Number of failed operations
        errors: List of error messages
    """
    success_count: int = 0
    error_count: int = 0
    errors: List[str] = field(default_factory=list)


def _compute_cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def _compute_euclidean_distance(a: List[float], b: List[float]) -> float:
    """Compute Euclidean distance between two vectors."""
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")

    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _compute_dot_product(a: List[float], b: List[float]) -> float:
    """Compute dot product of two vectors."""
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")

    return sum(x * y for x, y in zip(a, b))


class VectorDatabase:
    """
    Vector database with HNSW indexing and similarity search.

    This class provides a pure Python implementation with optional
    native acceleration when the ABI library is available.

    Example:
        >>> db = VectorDatabase(dimensions=384)
        >>> db.add([0.1, 0.2, ...], metadata={"label": "example"})
        >>> results = db.search([0.15, 0.25, ...], top_k=10)
        >>> for r in results:
        ...     print(f"ID: {r.id}, Score: {r.score:.4f}")

        >>> # Batch operations
        >>> db.add_batch([
        ...     {"vector": [0.1, ...], "metadata": {"label": "a"}},
        ...     {"vector": [0.2, ...], "metadata": {"label": "b"}},
        ... ])
    """

    def __init__(
        self,
        name: str = "default",
        dimensions: int = 0,
        config: Optional[DatabaseConfig] = None,
    ):
        """
        Initialize a vector database.

        Args:
            name: Database name
            dimensions: Vector dimensions (0 = auto-detect)
            config: Database configuration
        """
        self._name = name
        self._config = config or DatabaseConfig(dimensions=dimensions)
        self._dimensions = dimensions or self._config.dimensions
        self._vectors: List[Dict[str, Any]] = []
        self._next_id = 0
        self._lib = None

        # Try to load native library
        try:
            from . import _load_library
            self._lib = _load_library()
        except (ImportError, AttributeError):
            pass

    @property
    def name(self) -> str:
        """Get database name."""
        return self._name

    @property
    def dimensions(self) -> int:
        """Get vector dimensions."""
        return self._dimensions

    @property
    def count(self) -> int:
        """Get number of vectors in database."""
        return len(self._vectors)

    def add(
        self,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[int] = None,
    ) -> int:
        """
        Add a vector to the database.

        Args:
            vector: Vector data
            metadata: Optional metadata to associate
            id: Optional custom ID (auto-generated if not provided)

        Returns:
            Vector ID

        Raises:
            ValueError: If vector dimension doesn't match
        """
        # Auto-detect dimensions from first vector
        if self._dimensions == 0:
            self._dimensions = len(vector)
        elif len(vector) != self._dimensions:
            raise ValueError(
                f"Vector dimension {len(vector)} doesn't match database dimension {self._dimensions}"
            )

        vec_id = id if id is not None else self._next_id
        self._next_id = max(self._next_id, vec_id + 1)

        self._vectors.append({
            "id": vec_id,
            "vector": list(vector),
            "metadata": metadata or {},
        })

        return vec_id

    def add_batch(
        self,
        records: List[Dict[str, Any]],
    ) -> BatchResult:
        """
        Add multiple vectors in batch.

        Args:
            records: List of dicts with 'vector' and optional 'metadata', 'id'

        Returns:
            BatchResult with success/error counts

        Example:
            >>> result = db.add_batch([
            ...     {"vector": [0.1, 0.2, 0.3], "metadata": {"label": "a"}},
            ...     {"vector": [0.4, 0.5, 0.6], "metadata": {"label": "b"}},
            ... ])
        """
        result = BatchResult()

        for record in records:
            try:
                vector = record.get("vector")
                if vector is None:
                    raise ValueError("Record missing 'vector' field")

                self.add(
                    vector=vector,
                    metadata=record.get("metadata"),
                    id=record.get("id"),
                )
                result.success_count += 1
            except Exception as e:
                result.error_count += 1
                result.errors.append(str(e))

        return result

    def get(self, id: int) -> Optional[Dict[str, Any]]:
        """
        Get a vector by ID.

        Args:
            id: Vector ID

        Returns:
            Dict with 'id', 'vector', 'metadata' or None if not found
        """
        for entry in self._vectors:
            if entry["id"] == id:
                return entry.copy()
        return None

    def update(
        self,
        id: int,
        vector: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update a vector.

        Args:
            id: Vector ID to update
            vector: New vector data (optional)
            metadata: New metadata (optional)

        Returns:
            True if updated, False if not found
        """
        for entry in self._vectors:
            if entry["id"] == id:
                if vector is not None:
                    if len(vector) != self._dimensions:
                        raise ValueError(
                            f"Vector dimension {len(vector)} doesn't match database dimension {self._dimensions}"
                        )
                    entry["vector"] = list(vector)
                if metadata is not None:
                    entry["metadata"] = metadata
                return True
        return False

    def delete(self, id: int) -> bool:
        """
        Delete a vector by ID.

        Args:
            id: Vector ID to delete

        Returns:
            True if deleted, False if not found
        """
        for i, entry in enumerate(self._vectors):
            if entry["id"] == id:
                self._vectors.pop(i)
                return True
        return False

    def search(
        self,
        query: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        include_vectors: bool = False,
    ) -> List[SearchResult]:
        """
        Search for similar vectors.

        Args:
            query: Query vector
            top_k: Number of results to return
            filter: Metadata filter (optional)
            include_vectors: Include vector data in results

        Returns:
            List of SearchResult objects sorted by similarity

        Example:
            >>> results = db.search([0.1, 0.2, 0.3], top_k=5)
            >>> results = db.search(query, filter={"category": "news"})
        """
        if len(query) != self._dimensions:
            raise ValueError(
                f"Query dimension {len(query)} doesn't match database dimension {self._dimensions}"
            )

        # Compute similarities
        results = []
        for entry in self._vectors:
            # Apply metadata filter if provided
            if filter:
                if not self._matches_filter(entry["metadata"], filter):
                    continue

            # Compute similarity based on distance metric
            metric = self._config.distance_metric
            if metric == DistanceMetric.COSINE:
                score = _compute_cosine_similarity(query, entry["vector"])
            elif metric == DistanceMetric.EUCLIDEAN:
                # Convert distance to similarity (lower distance = higher similarity)
                distance = _compute_euclidean_distance(query, entry["vector"])
                score = 1.0 / (1.0 + distance)
            else:  # DOT_PRODUCT
                score = _compute_dot_product(query, entry["vector"])

            result = SearchResult(
                id=entry["id"],
                score=score,
                vector=entry["vector"] if include_vectors else None,
                metadata=entry["metadata"],
            )
            results.append(result)

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def hybrid_search(
        self,
        query_vector: List[float],
        query_text: Optional[str] = None,
        top_k: int = 10,
        alpha: float = 0.5,
    ) -> List[SearchResult]:
        """
        Hybrid search combining vector and text similarity.

        Args:
            query_vector: Query vector
            query_text: Query text for keyword matching
            top_k: Number of results to return
            alpha: Weight for vector similarity (0-1)

        Returns:
            List of SearchResult objects
        """
        # Get vector search results
        vector_results = self.search(query_vector, top_k=top_k * 2)

        if not query_text:
            return vector_results[:top_k]

        # Score text matches
        query_words = set(query_text.lower().split())
        for result in vector_results:
            text_score = 0.0
            metadata = result.metadata or {}

            # Check text fields in metadata
            for value in metadata.values():
                if isinstance(value, str):
                    value_words = set(value.lower().split())
                    overlap = len(query_words & value_words)
                    text_score = max(text_score, overlap / max(len(query_words), 1))

            # Combine scores
            result.score = alpha * result.score + (1 - alpha) * text_score

        # Re-sort and return
        vector_results.sort(key=lambda x: x.score, reverse=True)
        return vector_results[:top_k]

    def _matches_filter(self, metadata: Dict[str, Any], filter: Dict[str, Any]) -> bool:
        """Check if metadata matches filter conditions."""
        for key, value in filter.items():
            if key.startswith("$"):
                # Handle operators
                if key == "$and":
                    return all(self._matches_filter(metadata, f) for f in value)
                elif key == "$or":
                    return any(self._matches_filter(metadata, f) for f in value)
                elif key == "$not":
                    return not self._matches_filter(metadata, value)
            else:
                # Direct field match
                meta_value = metadata.get(key)
                if isinstance(value, dict):
                    # Handle comparison operators
                    for op, comp_value in value.items():
                        if op == "$eq" and meta_value != comp_value:
                            return False
                        elif op == "$ne" and meta_value == comp_value:
                            return False
                        elif op == "$gt" and not (meta_value is not None and meta_value > comp_value):
                            return False
                        elif op == "$gte" and not (meta_value is not None and meta_value >= comp_value):
                            return False
                        elif op == "$lt" and not (meta_value is not None and meta_value < comp_value):
                            return False
                        elif op == "$lte" and not (meta_value is not None and meta_value <= comp_value):
                            return False
                        elif op == "$in" and meta_value not in comp_value:
                            return False
                        elif op == "$nin" and meta_value in comp_value:
                            return False
                        elif op == "$exists" and (key in metadata) != comp_value:
                            return False
                elif meta_value != value:
                    return False
        return True

    def clear(self) -> None:
        """Clear all vectors from the database."""
        self._vectors.clear()
        self._next_id = 0

    def stats(self) -> DatabaseStats:
        """Get database statistics."""
        memory = sum(
            len(entry["vector"]) * 4 + len(json.dumps(entry["metadata"]))
            for entry in self._vectors
        )

        return DatabaseStats(
            vector_count=len(self._vectors),
            dimensions=self._dimensions,
            index_type=self._config.index_type.name.lower(),
            memory_usage_bytes=memory,
        )

    def optimize(self) -> None:
        """Optimize the database index for better search performance."""
        # In Python implementation, this is a no-op
        # Native implementation would rebuild HNSW index
        pass

    def save(self, path: Optional[str] = None) -> None:
        """
        Save database to file.

        Args:
            path: File path (uses config path if not provided)
        """
        path = path or self._config.path
        if path == ":memory:":
            raise ValueError("Cannot save in-memory database")

        data = {
            "name": self._name,
            "dimensions": self._dimensions,
            "config": {
                "distance_metric": self._config.distance_metric.name,
                "index_type": self._config.index_type.name,
            },
            "vectors": self._vectors,
            "next_id": self._next_id,
        }

        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "VectorDatabase":
        """
        Load database from file.

        Args:
            path: File path

        Returns:
            Loaded VectorDatabase instance
        """
        with open(path, "r") as f:
            data = json.load(f)

        config = DatabaseConfig(
            path=path,
            dimensions=data["dimensions"],
            distance_metric=DistanceMetric[data["config"]["distance_metric"]],
            index_type=IndexType[data["config"]["index_type"]],
        )

        db = cls(
            name=data["name"],
            dimensions=data["dimensions"],
            config=config,
        )

        db._vectors = data["vectors"]
        db._next_id = data.get("next_id", len(data["vectors"]))

        return db

    def __len__(self) -> int:
        return len(self._vectors)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self._vectors)

    def __contains__(self, id: int) -> bool:
        return any(entry["id"] == id for entry in self._vectors)


class DatabaseContext:
    """
    Database context for framework integration.

    This class wraps the VectorDatabase to provide a consistent interface
    with other ABI framework modules.

    Example:
        >>> from abi.config import DatabaseConfig
        >>> ctx = DatabaseContext(DatabaseConfig(path="./vectors.db"))
        >>> ctx.insert_vector(1, [0.1, 0.2, 0.3], {"label": "test"})
        >>> results = ctx.search_vectors([0.1, 0.2, 0.3], top_k=5)
    """

    def __init__(self, config: Optional["DatabaseConfig"] = None):
        """
        Initialize the database context.

        Args:
            config: Database configuration from abi.config
        """
        from .config import DatabaseConfig as AbiDatabaseConfig

        if config is None:
            config = AbiDatabaseConfig.defaults()

        db_config = DatabaseConfig(
            path=config.path,
            dimensions=config.dimensions,
            distance_metric=DistanceMetric[config.distance_metric.name],
            index_type=IndexType[config.index_type.name],
            cache_size=config.cache_size,
            wal_enabled=config.wal_enabled,
        )

        self._db = VectorDatabase(config=db_config)

    @property
    def database(self) -> VectorDatabase:
        """Get the underlying database."""
        return self._db

    def insert_vector(
        self,
        id: int,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert a vector into the database."""
        self._db.add(vector, metadata=metadata, id=id)

    def search_vectors(
        self,
        query: List[float],
        top_k: int = 10,
    ) -> List[SearchResult]:
        """Search for similar vectors."""
        return self._db.search(query, top_k=top_k)

    def get_stats(self) -> DatabaseStats:
        """Get database statistics."""
        return self._db.stats()

    def optimize(self) -> None:
        """Optimize the database index."""
        self._db.optimize()

    def close(self) -> None:
        """Close the database."""
        # Save if persistent
        if self._db._config.path != ":memory:":
            try:
                self._db.save()
            except Exception:
                pass


# Convenience functions

def create_database(
    name: str = "default",
    dimensions: int = 0,
    path: Optional[str] = None,
) -> VectorDatabase:
    """
    Create a new vector database.

    Args:
        name: Database name
        dimensions: Vector dimensions
        path: Optional file path

    Returns:
        VectorDatabase instance
    """
    config = DatabaseConfig(
        path=path or ":memory:",
        dimensions=dimensions,
    )
    return VectorDatabase(name=name, config=config)


def open_database(path: str) -> VectorDatabase:
    """
    Open an existing database from file.

    Args:
        path: Path to database file

    Returns:
        VectorDatabase instance
    """
    return VectorDatabase.load(path)


def is_enabled() -> bool:
    """Check if database features are available."""
    return True
