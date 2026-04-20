vllm serve nomic-ai/nomic-embed-text-v1.5 \
  --port 8102 \
  --gpu-memory-utilization 0.1 \
  --max-num-seqs 64