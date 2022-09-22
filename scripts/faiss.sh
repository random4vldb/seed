python -m pyserini.index.faiss \
  --input pyserini_dpr_$1 \
  --output pyserini_faiss_$1 \
  --hnsw \
  --pq