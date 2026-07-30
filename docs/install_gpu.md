# GPU Installation

This profile targets NVIDIA GPUs through CUDA. It is separate from the Ascend
NPU profile in `docs/install_npu.md`.

## Prerequisites

- Linux and Python 3.11.
- An NVIDIA driver compatible with CUDA 12.4.
- Conda, or a Python virtual environment with a CUDA-enabled PyTorch wheel.
- A compatible VERL checkout for formal distributed training.

Check the host before installing:

```bash
nvidia-smi
```

## Conda installation

The repository environment file installs the CUDA 12.4 PyTorch runtime and the
declared candidate profile (`torch==2.8.0`, `vllm==0.11.0`):

```bash
conda env create -f environment.yml
conda activate eviback-gpu
python -m pip install -e '.[test,data,retrieval]'
```

For a pip-only setup, select the CUDA wheel index explicitly:

```bash
python3.11 -m venv .venv-gpu
. .venv-gpu/bin/activate
python -m pip install --upgrade pip
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements-gpu.txt
python -m pip install -e '.[test,data,retrieval]'
```

Verify the accelerator and the package:

```bash
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda)"
python -m pytest -q tests
```

For formal training, install the compatible VERL checkout, apply
`third_party/verl/group_postnorm.patch`, and then install the training extra:

```bash
python -m pip install -e '.[train]'
bash scripts/train_actor.sh --config configs/training/qwen3_1.7b.yaml
```

The exact formal PyTorch and Ray builds remain a release blocker. Treat the
versions in this document as a reproducible GPU installation candidate, not as
a claim that all historical runs used these exact builds. GPU end-to-end
training has not been validated on this host.

The retriever uses CUDA when started with:

```bash
bash scripts/start_retriever.sh \
  --index /path/to/e5_Flat.index \
  --corpus /path/to/docs.jsonl \
  --model intfloat/e5-base-v2 \
  --device cuda
```