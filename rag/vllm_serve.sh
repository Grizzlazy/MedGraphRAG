# vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
#     --port 8101 \
#     --quantization awq \
#     --gpu-memory-utilization 0.8 \
#     --max-num-seqs 16

vllm serve meta-llama/Llama-3.2-3B-Instruct \
  --port 8101 \
  --gpu-memory-utilization 0.8 \
  --max-num-seqs 16 \
  --max-model-len 8192

# vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
#   --port 8101 \
#   --gpu-memory-utilization 0.8 \
#   --max-num-seqs 16 \
#   --max-model-len 8192