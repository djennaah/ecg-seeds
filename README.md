# ecg-seeds: do SSL rankings for ECG hold across random seeds?

Fork of [ecg-fix](https://github.com/zackeberger/ecg-fix) (ICML 2026). The model code and the benchmark are
theirs; we add **multiple seeds**, **extra metrics** and **paired significance tests**.

## Research question
Does the ranking of ECG self-supervised methods reported by MERL hold under a stricter evaluation
(several seeds, AUPRC as a second metric, significance tests)? In particular, does a random-init encoder
really match pretrained ones?

## What we ran
- Task: PTB-XL super-classes, linear probing with 1% / 10% / 100% of the labels, macro AUROC and AUPRC.
- Models: MERL, CLOCS (pretrained, frozen) and the official Random baseline
  (`Random_500Hz_Z_score_none_No_Bandpass_Resnet18`).
- Seeds 42-46 for each model (15 runs, 3 label fractions each).

## Main results (macro AUROC, mean ± sd over 5 seeds)
| Labels | MERL | CLOCS | Random init |
|---|---|---|---|
| 1% | 0.826 ± 0.005 | 0.759 ± 0.007 | 0.727 ± 0.077 |
| 10% | 0.881 ± 0.002 | 0.805 ± 0.002 | 0.829 ± 0.004 |
| 100% | 0.901 ± 0.000 | 0.821 ± 0.000 | 0.861 ± 0.004 |

- MERL beats Random init at every label fraction (5/5 seeds, significant after Holm correction).
- Random init beats CLOCS at 10% and 100% labels (significant). At 1% the CLOCS vs Random comparison is
  not settled: one Random seed (45) gives 0.590 AUROC while the other four are between 0.750 and 0.770.
  That seed is kept in all results; its cause is not identified.
- Seeds matter at 1% and 10% labels; at 100% labels the seed sd of MERL and CLOCS is about 0.000.

Full tables, tests and figures: `analysis/` (`summary.md`, `tests.csv`, `ranking.csv`, `fig1_*`, `fig2_*`).

## Important notes
- Encoders of MERL and CLOCS are frozen, so embeddings do not depend on the seed. We reuse the seed-42
  embeddings for the other seeds. The seed changes only the label subsample and the logistic-regression classifier.
- For Random, the seed also changes the encoder initialisation, so everything is recomputed for each seed.
- Seeds 42-44 of Random were run with an earlier version of `run_random_seeds.sh`, seed 45 with a first
  version that did not stop on errors, seed 46 with the current one.
- To run on Python 3.13 we relaxed `requirements.txt`, patched one import in
  `external_src/D_BETA/models/cross_layer.py`, and set `preprocess_workers` to 2 in `config.json` (RAM).

## Reproduce
1. `pip install -r requirements.txt`
2. Download PTB-XL 1.0.3 (500 Hz) and the weights `doprakah/ecg-fix-weights` (see ecg-fix).
3. Seed 42: `python -m src.main --datasets PTBXL_super --models MERL CLOCS`
4. More seeds: `run_clocs_merl_seeds.sh` (MERL, CLOCS) and `run_random_seeds.sh 42 43 44 45 46` (Random).
5. Analysis from the saved predictions (no GPU, about 2 min): `python analyze.py`

## Layout of `results/`
`results/<model>/seed<N>/<label fraction>/` with `<model>.pkl` (test predictions), `<model>_metric.pkl`
(metrics) and `<model>_dataset.pkl` (counts). Models: `clocs`, `merl`, `random_init`; fractions `0.01`, `0.1`, `1.0`.

## Authors
Djenna Hammache, Wafa Abed. Stockholm University.
