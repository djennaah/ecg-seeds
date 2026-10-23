#!/bin/bash
# Run MERL and CLOCS for several seeds. Seed 42 must already be done (data/results).
# Frozen encoders give the same embeddings for any seed, so the cached ones are reused.
# Seed 43 was started by hand with the same commands, then this loop ran seeds 44-46.
cd /content/ecg-seeds
archive_current() {
  CUR=$(grep -oE '"seed": *[0-9]+' config.json | grep -oE '[0-9]+$')
  [ -d data/results ] && [ ! -d results_seed$CUR ] && mv data/results results_seed$CUR
  [ -d metrics ] && [ ! -d metrics_seed$CUR ] && mv metrics metrics_seed$CUR
}
for seed in 43 44 45 46; do
  archive_current
  sed -i -E "s/\"seed\": *[0-9]+/\"seed\": $seed/" config.json
  for f in data/embeddings/PTBXL_super/*/MERL_done.json data/embeddings/PTBXL_super/*/CLOCS_done.json; do
    sed -i -E "s/\"seed\": *[0-9]+/\"seed\": $seed/" "$f"
  done
  python -m src.main --datasets PTBXL_super --models MERL CLOCS > run_merl_clocs_seed$seed.log 2>&1
  echo "seed $seed finished at $(date)" >> seeds_progress_merl_clocs.log
done
archive_current
