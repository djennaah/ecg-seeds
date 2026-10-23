#!/bin/bash
# Run the official Random baseline (Random_500Hz_Z_score_none_No_Bandpass_Resnet18) for several seeds.
#
# Usage: bash run_random_seeds.sh 42 43 44 45 46
#
# MERL and CLOCS are frozen public checkpoints, so their embeddings are the same for every seed and
# run_clocs_merl_seeds.sh reuses the cached ones. The Random encoder is initialised from the seed,
# so here the cached embeddings are deleted before each seed and everything is recomputed.
# The seed also changes the label subsample and the linear probe.
#
# Needs: PTB-XL 1.0.3 (500 Hz) already downloaded (see README). On CPU one seed takes about 1 h.
# Output (same layout as results/clocs_merl): results/random/seed<N>/{results,metrics,config.json}
# Optional: BACKUP_DIR=/path bash run_random_seeds.sh 45 46   also copies each seed to /path.
# Note: seeds 42-44 were run with an earlier version of this loop, seed 45 with a first version
# that did not stop on errors, seed 46 with this version. The procedure is the same.

REPO=${REPO:-/content/ecg-seeds}
MODEL=Random_500Hz_Z_score_none_No_Bandpass_Resnet18
cd "$REPO" || exit 1

if [ $# -eq 0 ]; then echo "Usage: bash run_random_seeds.sh SEED [SEED ...]"; exit 1; fi
# Never start a second pipeline on top of a running one
if pgrep -f "[s]rc.main" > /dev/null; then echo "A run is already active, stop."; exit 1; fi

for seed in "$@"; do
  echo "=== seed $seed started $(date)" | tee -a seeds_progress_random.log

  # Start from clean folders so seeds never mix (embeddings depend on the seed for Random)
  rm -rf data/results metrics data/embeddings/PTBXL_super

  # Set the seed in config.json
  sed -i -E "s/\"seed\": *[0-9]+/\"seed\": $seed/" config.json

  python -m src.main --datasets PTBXL_super --models $MODEL > run_random_seed$seed.log 2>&1
  code=$?

  # Stop at the first failure instead of saving empty folders
  if [ $code -ne 0 ] || [ ! -d data/results ]; then
    echo "=== seed $seed FAILED (code $code) $(date)" | tee -a seeds_progress_random.log
    tail -n 20 run_random_seed$seed.log
    exit 1
  fi

  # Archive this seed
  out=results/random/seed$seed
  rm -rf "$out"; mkdir -p "$out"
  mv data/results "$out/results"
  mv metrics "$out/metrics" 2>/dev/null
  cp config.json "$out/"

  if [ -n "$BACKUP_DIR" ]; then
    mkdir -p "$BACKUP_DIR/random" && cp -r "$out" "$BACKUP_DIR/random/" && echo "saved to $BACKUP_DIR"
  fi

  echo "=== seed $seed done $(date)" | tee -a seeds_progress_random.log
done
echo "ALL DONE $(date)" | tee -a seeds_progress_random.log
