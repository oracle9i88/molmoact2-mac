#!/usr/bin/env python3
"""One-time preparation of a downloaded lerobot/MolmoAct2-SO100_101-LeRobot checkpoint.

usage: prepare.py CKPT_DIR BASE_DIR

1. Fix CKPT_DIR/config.json for lerobot 0.7.0. The published config still carries four
   fields that were renamed/removed (enable_lora_vlm, enable_lora_action_expert,
   train_action_expert_only, model_dtype), so loading fails with a draccus DecodingError.
   The original is kept as config.orig.json.

2. Build BASE_DIR/model.safetensors from CKPT_DIR/model.safetensors.
   The LeRobot policy first instantiates the base model `allenai/MolmoAct2-SO100_101`
   (another 21.8 GB) and then overwrites every tensor with the LeRobot weights. The two
   files hold the same 1295 tensors; the LeRobot names just have a "model." prefix. So we
   rewrite only the safetensors header and stream the tensor data through unchanged.
   Nothing is loaded into RAM. BASE_DIR must already contain the small files of the base
   repo (config.json, tokenizer*.json, processor_config.json, norm_stats.json, ...).
"""
import json
import os
import shutil
import struct
import sys

ckpt, base = sys.argv[1], sys.argv[2]

# --- 1. config.json -------------------------------------------------------------------
cfg_path = os.path.join(ckpt, "config.json")
cfg = json.load(open(cfg_path))
stale = ["enable_lora_vlm", "enable_lora_action_expert", "train_action_expert_only", "model_dtype"]
if any(k in cfg for k in stale):
    shutil.copy(cfg_path, os.path.join(ckpt, "config.orig.json"))
    for k in stale:
        cfg.pop(k, None)
    cfg["train_mode_vlm"] = "fft"  # no LoRA adapters: the checkpoint stores full weights
    json.dump(cfg, open(cfg_path, "w"), indent=4)
    print("config.json patched (original saved as config.orig.json)")
else:
    print("config.json already compatible")

# --- 2. base weights ------------------------------------------------------------------
for need in ("config.json", "tokenizer.json", "processor_config.json", "norm_stats.json"):
    if not os.path.exists(os.path.join(base, need)):
        sys.exit(f"{base}/{need} missing - download the small files of allenai/MolmoAct2-SO100_101 first")

src = os.path.join(ckpt, "model.safetensors")
dst = os.path.join(base, "model.safetensors")
if os.path.exists(dst):
    print("base model.safetensors already there")
else:
    with open(src, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        meta = header.pop("__metadata__", None)
        new = {}
        for k, v in header.items():
            if not k.startswith("model."):
                sys.exit(f"unexpected tensor name: {k}")
            new[k[len("model."):]] = v
        if meta is not None:
            new["__metadata__"] = meta
        hb = json.dumps(new, separators=(",", ":")).encode()
        hb += b" " * (-len(hb) % 8)
        with open(dst + ".tmp", "wb") as out:
            out.write(struct.pack("<Q", len(hb)))
            out.write(hb)
            while buf := f.read(64 << 20):
                out.write(buf)
    os.replace(dst + ".tmp", dst)
    print(f"wrote {dst}: {os.path.getsize(dst) / 1e9:.2f} GB, {len(header)} tensors")

# The loader prefers a shard index if one exists; we have a single file instead.
idx = os.path.join(base, "model.safetensors.index.json")
if os.path.exists(idx):
    os.replace(idx, idx + ".unused")
