#!/usr/bin/env python3
"""Open-loop replay on recorded SO-100/101 episodes: no robot needed.

usage: replay_test.py --ckpt CKPT_DIR --base BASE_DIR --data DATA_DIR [--episodes 000000 ...]

Every --stride frames, feed the recorded camera frames + joint state to the model, predict
one 30-step chunk, and compare it with the recorded action[t:t+30] (degrees).
Baseline "hold" = repeat the current joint state for 30 steps. A working pipeline should be
far below the baseline; a broken one (wrong image format, joint frame, normalisation, or a
bad dtype/backend) will not be.

DATA_DIR layout is what fetch_episodes.sh produces (LeRobot v2.1 dataset, old SO-100
calibration frame). That frame equals the model frame; the LeRobot checkpoint's processors
convert from the *new* LeRobot calibration frame, so the data is mapped into the new frame
first and then goes through the unmodified pipeline. This also checks the frame conversion.
"""
import argparse
import json
import time

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--base", required=True)
p.add_argument("--data", required=True)
p.add_argument("--episodes", nargs="+", default=["000000", "000013", "000027"])
p.add_argument("--stride", type=int, default=30)
p.add_argument("--device", default="mps")
p.add_argument("--task", default=None)
p.add_argument("--out", default=None)
a = p.parse_args()

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from molmoact2_mac import load

policy, pre, post, cfg = load(a.ckpt, a.base, a.device)
signs = np.array(cfg.joint_signs, dtype=np.float32)
offs = np.array(cfg.joint_offsets, dtype=np.float32)
to_lerobot = lambda x: signs * (x - offs)  # model frame -> new LeRobot frame
task = a.task or json.loads(open(f"{a.data}/meta/tasks.jsonl").readline())["task"]
print("task:", task, flush=True)
H = cfg.n_action_steps


def frames(path):
    cap, out = cv2.VideoCapture(path), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return out


def img(x):
    return torch.from_numpy(x).permute(2, 0, 1).float() / 255.0


names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
all_m, all_b, lat = [], [], []
g = torch.Generator().manual_seed(0)
for ep in a.episodes:
    d = f"{a.data}/ep{ep}"
    t = pq.read_table(f"{d}/data.parquet").to_pydict()
    state = to_lerobot(np.array(t["observation.state"], dtype=np.float32))
    action = to_lerobot(np.array(t["action"], dtype=np.float32))
    top, side = frames(f"{d}/realsense_top.mp4"), frames(f"{d}/realsense_side.mp4")
    n = min(len(state), len(top), len(side))
    em, eb = [], []
    for s in range(0, n - H, a.stride):
        obs = {
            "observation.images.cam0": img(top[s]),
            "observation.images.cam1": img(side[s]),
            "observation.state": torch.from_numpy(state[s].copy()),
            "task": task,
        }
        noise = torch.randn(1, cfg.chunk_size, cfg.expected_max_action_dim, generator=g).to(a.device)
        t0 = time.time()
        pred = post(policy.predict_action_chunk(pre(obs), noise=noise)).float().cpu().numpy()[0]
        lat.append(time.time() - t0)
        gt = action[s : s + H]
        em.append(np.abs(pred - gt))
        eb.append(np.abs(state[s][None] - gt))
    em, eb = np.stack(em), np.stack(eb)
    print(f"episode {ep}: {len(em)} chunks, model MAE {em.mean():.2f} deg, hold baseline {eb.mean():.2f} deg", flush=True)
    all_m.append(em)
    all_b.append(eb)

m, b = np.concatenate(all_m), np.concatenate(all_b)
print(f"\n{len(m)} chunks x {H} steps")
print(f"model MAE {m.mean():.2f} deg | hold baseline {b.mean():.2f} deg | ratio {m.mean() / b.mean():.2f}")
print("per joint (model/hold): " + "  ".join(f"{nm} {m[..., j].mean():.1f}/{b[..., j].mean():.1f}" for j, nm in enumerate(names)))
print("per step  (model/hold): " + "  ".join(f"#{k + 1} {m[:, k].mean():.1f}/{b[:, k].mean():.1f}" for k in (0, 4, 9, 14, 19, 29)))
print(f"latency per chunk incl. pre/post-processing: {np.mean(lat[1:]):.3f}s")
if a.out:
    json.dump(
        {"task": task, "model_mae": float(m.mean()), "hold_mae": float(b.mean()),
         "by_joint_model": m.mean(axis=(0, 1)).tolist(), "by_joint_hold": b.mean(axis=(0, 1)).tolist(),
         "by_step_model": m.mean(axis=(0, 2)).tolist(), "by_step_hold": b.mean(axis=(0, 2)).tolist(),
         "latency_s": float(np.mean(lat[1:]))},
        open(a.out, "w"), indent=1)
