#!/usr/bin/env bash

# Kill previous instances
pkill -f "vllm|ngrok|livekit-server|speech_to_text_streaming_infer_rnnt|agent.py"

# Launch vLLM
nohup vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --speculative-config '{"method": "ngram", "num_speculative_tokens": 6, "prompt_lookup_max": 4, "prompt_lookup_min": 1}' \
  --host 0.0.0.0 \
  --port 8000 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --dtype bfloat16 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 32768 \
  --max-model-len 65536 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --gpu-memory-utilization 0.85 \
  > /tmp/vllm.log 2>&1 &

# Launch NeMo
#HYDRA_FULL_ERROR=1 nohup uv run python ./speech_to_text_streaming_infer_rnnt.py \
#  pretrained_name="nvidia/parakeet-tdt-0.6b-v3" \
#  compute_dtype="float16" \
#  decoding.greedy.use_cuda_graph_decoder=False \
#  model_path=null \
#  dataset_manifest="/content/input_manifest.json" \
#  +output_filename="transcription_results.json" \
#  batch_size=32 \
#  chunk_secs=2.0 \
#  left_context_secs=10.0 \
#  right_context_secs=2.0 \
#  clean_groundtruth_text=False \
#  > /tmp/nemo.log 2>&1 &

# Launch LiveKit
nohup livekit-server --dev --bind 0.0.0.0 \
  > /tmp/livekit.log 2>&1 &

# Launch your agent
nohup bash -c 'cd demo16 && uv run python src/agent.py dev' \
  > /tmp/agent.log 2>&1 &

# Expose via ngrok
nohup ngrok http --url=unsuperfluously-consentaneous-fletcher.ngrok-free.app 3000 \
  > /tmp/ngrok.log 2>&1 &
