#!/bin/bash
# Download three episodes (~35 MB) of Beegbrain/pick_lemon_and_drop_in_bowl for replay_test.py.
# usage: fetch_episodes.sh DATA_DIR        (set HF_ENDPOINT to use a mirror)
set -e
D="${1:?usage: fetch_episodes.sh DATA_DIR}"
B="${HF_ENDPOINT:-https://huggingface.co}/datasets/Beegbrain/pick_lemon_and_drop_in_bowl/resolve/main"
mkdir -p "$D/meta"
for f in info.json tasks.jsonl episodes.jsonl; do
  curl -sS -L --fail "$B/meta/$f" -o "$D/meta/$f"
done
for e in 000000 000013 000027; do
  mkdir -p "$D/ep$e"
  curl -sS -L --fail "$B/data/chunk-000/episode_$e.parquet" -o "$D/ep$e/data.parquet"
  for c in realsense_top realsense_side; do
    curl -sS -L --fail "$B/videos/chunk-000/observation.images.$c/episode_$e.mp4" -o "$D/ep$e/$c.mp4"
  done
done
echo "done: $D"
