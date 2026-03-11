The repository is organized for clarity and maintainability:

prepare.py: A utility to download and tokenize a dataset into a memory-mapped binary format for efficient loading.

model.py: The heart of the project. Contains the complete definition of the Transformer architecture, including all layers like MoE, GQA, etc.

train.py: The main script for launching a distributed training job using FSDP.

sample.py: A multi-GPU, FSDP-aware script for generating text from a trained checkpoint.

export_to_safetensors.py: The script to convert internal training checkpoints to a Hugging Face-compatible format.
