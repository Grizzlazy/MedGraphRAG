vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
    --port 8101 \
    --quantization awq \
    --gpu-memory-utilization 0.8 \
    --max-num-seqs 16