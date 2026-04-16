# Vector DB Optimization

This tutorial outlines strategies to optimize vector database usage for retrieval-augmented generation.

## Topics
- Choosing embeddings dimensionality
- Index type selection (HNSW, IVF, PQ)
- Batch insertion and compaction
- Metrics to monitor (recall, latency)

## Example checklist
- [ ] Use normalized vectors
- [ ] Tune HNSW parameters (M, efConstruction)
- [ ] Run periodic index rebuilds for large insert volumes
