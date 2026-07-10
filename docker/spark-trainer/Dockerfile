# Training image for the DGX Spark (aarch64, GB10). Base: quant-tools:v3,
# the Spark-local image that already carries NVIDIA's ARM torch build
# (2.11 nv26.02) + transformers. bitsandbytes is deliberately absent: it
# ships no CUDA aarch64 binary, so training runs plain bf16 LoRA
# (qlora_train.py --quant none).
FROM quant-tools:v3
RUN pip install --no-cache-dir "peft>=0.19" "trl>=1.8" "datasets>=4.8"
