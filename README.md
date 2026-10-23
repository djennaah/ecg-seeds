# ecg-seeds: do SSL rankings for ECG hold across random seeds?

Fork of [ecg-fix](https://github.com/zackeberger/ecg-fix) (ICML 2026). All model code and benchmark
are theirs. We add **multiple seeds** and (soon) **statistical tests**.

## Research question
Does the ranking of ECG self-supervised methods reported by MERL hold under a stricter evaluation
(several seeds, extra metrics, significance tests)? In particular, does a random-init encoder
really match pretrained ones?

## What we ran
- Task: PTB-XL super-classes, linear probing at 1% / 10% / 100% of labels, macro AUROC / AUPRC.
- MERL and CLOCS (pretrained): seeds 42-46.
- Official Random baseline `Random_500Hz_Z_score_none_No_Bandpass_Resnet18`: seeds 42-46 (in progress).

## Important notes
- Encoders of MERL and CLOCS are frozen, so embeddings do not depend on the seed. We reuse the
  seed-42 embeddings for the other seeds. The seed only changes label subsampling and the
  logistic-regression classifier.
- For Random, the seed also changes the encoder initialisation, so everything is recomputed.
- Small changes to run on Python 3.13: relaxed versions in `requirements.txt` and one import in
  `external_src/D_BETA/models/cross_layer.py`; `preprocess_workers` set to 2 in `config.json` (RAM).

## Reproduce
1. Install: `pip install -r requirements.txt`
2. Download PTB-XL 1.0.3 (500 Hz) and the weights `doprakah/ecg-fix-weights` (see ecg-fix).
3. Seed 42: `python -m src.main --datasets PTBXL_super --models MERL CLOCS`
4. Other seeds: `run_clocs_merl_seeds.sh` (edit the seed list).

## Results
Layout: `results/<model>/seed<N>/results` (raw `.pkl`, predictions) and `results/<model>/seed<N>/metrics` (tables).
`run_clocs_merl_seeds.sh` writes `results_seedN` / `metrics_seedN` at the repo root; we moved them into `results/clocs_merl/seed<N>/` afterwards.
Final tables, tests and figures: to be added.

## Authors
Djenna Hammache, Wafa Abed. Stockholm University.
