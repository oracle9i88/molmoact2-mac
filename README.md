# MolmoAct2 on a Mac (Apple Silicon, PyTorch MPS)

[中文說明](README.zh.md)

The MolmoAct2 docs say an NVIDIA GPU is required. It isn't. The SO-100/101 checkpoint
[`lerobot/MolmoAct2-SO100_101-LeRobot`](https://huggingface.co/lerobot/MolmoAct2-SO100_101-LeRobot)
runs on a Mac GPU with **no change to the model code**: 0.82 s per 30-step action chunk,
bit-for-bit the same output as the CPU float32 reference.

This repo is not a port. It is one page of instructions, two things you must get right, and
small scripts to prepare the checkpoint and to verify your own machine.

## Measured

Mac Studio, M4 Max, 64 GB, macOS 27.2.1 · Python 3.12 · lerobot 0.7.0 · torch 2.15.0 · transformers 5.22.2

Synthetic fixed input (`smoke_test.py`), one 30-step chunk, warm:

| device / dtype | latency | diff vs CPU float32 (deg) | GPU memory |
|---|---|---|---|
| **MPS float32** | **0.82 s** | **max 0.0000, mean 0.0000** | 20.5 GiB |
| MPS bfloat16 | 0.80 s | max 26.5, mean 9.8 — **wrong** | 11.3 GiB |
| CPU float32 | 3.1 s | reference | – |
| CPU bfloat16 | 15.5 s | max 0.30, mean 0.06 | – |

Open-loop replay on real recordings (`replay_test.py`, 3 episodes of
`Beegbrain/pick_lemon_and_drop_in_bowl`, 36 chunks × 30 steps, MPS float32):

| | model | "hold still" baseline |
|---|---|---|
| mean abs error, all joints | 2.13° | 11.00° |
| step 1 / step 30 | 1.2° / 3.0° | 4.0° / 17.0° |
| latency incl. pre/post-processing | 0.84 s | |

That dataset is the sample used on the base model card and is very likely part of the
training mixture. The replay shows the whole pipeline is wired correctly on MPS (image
format, joint-frame conversion, normalisation, backend). It says nothing about how the
model generalises to your table.

## The two things you must get right

**1. Use float32 on MPS. Do not use bfloat16.**
bfloat16 is the default dtype in the LeRobot config. On MPS it runs without any error or
warning and returns actions that are off by ~10° on average (26° worst case) for the same
input. The same bfloat16 path on CPU is fine (0.06°), so this is an MPS backend problem, not
a precision problem. We have not tracked down which op. It is not faster either. On a real
arm this would look like "the model is bad" rather than "the numbers are wrong".

**2. The published `config.json` does not load with lerobot 0.7.0.**
It carries four fields that have since been renamed or removed, and loading stops with a
`draccus.utils.DecodingError`. `prepare.py` fixes the file and keeps the original.

Also useful: **you do not need to download the base model.** The LeRobot policy first loads
`allenai/MolmoAct2-SO100_101` (another 21.8 GB) and then overwrites every tensor with the
LeRobot weights. Both hold the same 1295 tensors, differing only by a `model.` name prefix.
`prepare.py` builds the base file locally by rewriting the safetensors header.

## Requirements

- Apple Silicon Mac. Measured on 64 GB. GPU memory in use is ~21 GiB, but process memory
  peaks at ~41 GiB **while loading** (the weights pass through RAM more than once). We have
  not tested 32/36/48 GB machines; expect heavy swapping or failure on 32 GB.
- ~45 GB free disk: 21.8 GB checkpoint + 21.8 GB generated base file.

## Steps

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install "lerobot[molmoact2]==0.7.0" pyarrow opencv-python
```

```bash
hf download lerobot/MolmoAct2-SO100_101-LeRobot --local-dir models/ckpt
```

```bash
hf download allenai/MolmoAct2-SO100_101 --local-dir models/base --exclude "*.safetensors" "*.py"
```

```bash
python prepare.py models/ckpt models/base
```

```bash
python smoke_test.py --ckpt models/ckpt --base models/base
```

Expected on the synthetic input: step 1 = `[4.78, -65.89, 70.87, 38.89, -0.25, 8.71]`.
If you see different numbers, something is wrong with your setup.

Optional, real-data replay (~35 MB download):

```bash
./fetch_episodes.sh data/pick_lemon
```

```bash
python replay_test.py --ckpt models/ckpt --base models/base --data data/pick_lemon
```

To check any other backend or dtype against the CPU reference:

```bash
python smoke_test.py --ckpt models/ckpt --base models/base --device cpu --runs 1 --save ref.pt
```

```bash
python smoke_test.py --ckpt models/ckpt --base models/base --device mps --ref ref.pt
```

## Using it in your own code

```python
from molmoact2_mac import load

policy, pre, post, cfg = load("models/ckpt", "models/base")        # MPS, float32
obs = {
    "observation.images.cam0": img0,      # float tensor, CHW, 0..1
    "observation.images.cam1": img1,
    "observation.state": joints_deg,      # 6 joint angles, LeRobot SO-101 calibration, degrees
    "task": "pick the yellow lemon and drop it in the red bowl",
}
actions = post(policy.predict_action_chunk(pre(obs)))   # (1, 30, 6), absolute joint targets, degrees
```

What `load()` does differently from the defaults: `dtype=float32`,
`enable_inference_cuda_graph=False`, and `checkpoint_path` pointed at the local base
directory (in the policy config and in the saved preprocessor).

## Not covered

- No test on a physical arm yet, and no closed-loop evaluation.
- No test on data the model has not seen.
- Root cause of the MPS bfloat16 error is unknown.
- Fine-tuning on a Mac was not tried.

Before running any learned policy on hardware: limit speed and workspace outside the model,
keep an emergency stop within reach, and follow the safety section of the model card.

## Licence and credits

Scripts here: Apache-2.0. No weights are redistributed.
MolmoAct2 (code and weights) is by the Allen Institute for AI, Apache-2.0, intended for
research and educational use under Ai2's Responsible Use Guidelines:
[allenai/molmoact2](https://github.com/allenai/molmoact2).
LeRobot is by Hugging Face, Apache-2.0. The replay data is
[Beegbrain/pick_lemon_and_drop_in_bowl](https://huggingface.co/datasets/Beegbrain/pick_lemon_and_drop_in_bowl).
