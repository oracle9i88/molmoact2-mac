#!/usr/bin/env python3
"""Load MolmoAct2 and time action-chunk prediction on a fixed synthetic input.

usage: smoke_test.py --ckpt CKPT_DIR --base BASE_DIR [--device mps|cpu] [--dtype float32|bfloat16]
                     [--runs 5] [--save out.pt] [--ref ref.pt]

The input (random images, fixed joint angles, fixed flow-matching start noise) is seeded,
so results from different devices / dtypes can be compared with --save / --ref.
"""
import argparse
import resource
import time

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--base", required=True)
p.add_argument("--device", default="mps")
p.add_argument("--dtype", default="float32")
p.add_argument("--runs", type=int, default=5)
p.add_argument("--save")
p.add_argument("--ref")
p.add_argument("--task", default="pick up the red cube and put it in the box")
a = p.parse_args()

import torch
from molmoact2_mac import load, sync

t0 = time.time()
policy, pre, post, cfg = load(a.ckpt, a.base, a.device, getattr(torch, a.dtype))
sync(a.device)
print(f"loaded in {time.time() - t0:.1f}s  device={a.device} dtype={a.dtype}", flush=True)

g = torch.Generator().manual_seed(0)
obs = {
    "observation.images.cam0": torch.rand(3, 224, 224, generator=g),
    "observation.images.cam1": torch.rand(3, 224, 224, generator=g),
    "observation.state": torch.tensor([5.0, -60.0, 70.0, 40.0, 0.0, 20.0]),
    "task": a.task,
}
batch = pre(obs)
noise = torch.randn(1, cfg.chunk_size, cfg.expected_max_action_dim, generator=g).to(a.device)

times = []
for i in range(a.runs):
    sync(a.device)
    t = time.time()
    act = policy.predict_action_chunk(batch, noise=noise)
    sync(a.device)
    times.append(time.time() - t)
    print(f"run {i + 1}: {times[-1]:.3f}s", flush=True)
out = post(act).float().cpu()
print("action shape", tuple(out.shape), "| any NaN:", bool(torch.isnan(out).any()))
print("step 1 ", [round(x, 2) for x in out[0, 0].tolist()])
print("step 30", [round(x, 2) for x in out[0, -1].tolist()])
warm = times[1:] or times
print(f"latency per {cfg.n_action_steps}-step chunk: first {times[0]:.3f}s, then mean {sum(warm) / len(warm):.3f}s")
if a.device == "mps":
    print(f"MPS memory: allocated {torch.mps.current_allocated_memory() / 2**30:.1f} GiB, "
          f"driver {torch.mps.driver_allocated_memory() / 2**30:.1f} GiB")
print(f"peak process RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.1f} GiB")
if a.save:
    torch.save(out, a.save)
if a.ref:
    d = (out - torch.load(a.ref)).abs()
    print(f"vs reference: max {d.max():.4f} deg, mean {d.mean():.4f} deg")
