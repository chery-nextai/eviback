# Ascend NPU Installation

This profile targets Ascend 910B through `torch-npu`. It must not be combined
with the CUDA PyTorch or CUDA vLLM installation described in
`docs/install_gpu.md`.

## Prerequisites

- An Ascend 910B host with a working driver and `npu-smi`.
- CANN Toolkit and the matching ATB runtime installed by the system operator.
- Linux, Python 3.11, and Conda.

The profile was smoke-tested locally with CANN 8.5.2, `torch==2.8.0`,
`torch-npu==2.8.0.post4`, and `ray==2.56.0`. This verifies the base NPU runtime,
not a general vLLM/VERL compatibility guarantee. Other CANN and wheel
combinations must be checked against the Ascend compatibility matrix before use.

Check the host and source the system runtime. Adjust the path for the local
CANN installation:

```bash
npu-smi info
export ASCEND_CANN_PATH=${ASCEND_CANN_PATH:-/usr/local/Ascend/cann-8.5.2}
source "${ASCEND_CANN_PATH}/set_env.sh"
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
  source /usr/local/Ascend/nnal/atb/set_env.sh
fi
```

## Installation

Create the base environment, then install the matched NPU wheels and EviBack:

```bash
conda env create -f environment-npu.yml
conda activate eviback-npu
python -m pip install --upgrade pip
python -m pip install -r requirements-npu.txt
python -m pip install -e '.[test,data,retrieval]'
```

If `torch-npu` is distributed through an Ascend or institutional package index,
configure that index before installing `requirements-npu.txt`. Do not replace
`torch-npu` with a CUDA build of PyTorch. The wheel, CANN, driver, and Python
versions must remain compatible.

Verify the NPU runtime and the package:

```bash
python -c "import torch, torch_npu; assert hasattr(torch, 'npu') and torch.npu.is_available(); print(torch.__version__, torch_npu.__version__)"
python -m pytest -q tests
```

The retriever uses NPU when started with:

```bash
bash scripts/start_retriever.sh \
  --index /path/to/e5_Flat.index \
  --corpus /path/to/docs.jsonl \
  --model intfloat/e5-base-v2 \
  --device npu
```

Formal NPU training additionally requires an Ascend-compatible
`vllm-ascend`/VERL stack and the repository's VERL patch. Those components are
not installed by `requirements-npu.txt` or the CUDA-oriented `[train]` extra.
Install a complete matched stack before launching distributed training. For
example, the vLLM Ascend 0.13.0 compatibility profile pairs vLLM 0.13.0 with
CANN 8.5.0 and PyTorch 2.8.0 / torch-npu 2.8.0.post2; the local smoke profile
uses different CANN and torch-npu patch versions and must not be mixed with it
without revalidation.

After installing the matched training stack and applying the VERL patch, start
training directly with:

```bash
bash scripts/train_actor.sh --config configs/training/qwen3_1.7b.yaml
```