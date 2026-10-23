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
## Contributions
- **Djenna Hammache**: ran MERL and CLOCS (seeds 42-46) on PTB-XL (data download and preprocessing),
  organised the repository and its results layout, ran the analysis and wrote this README.
- **Wafa Abed**: downloaded the pretrained weights, ran the official Random baseline (seeds 42-46) and
  wrote `run_random_seeds.sh`.
- **Both**: chose the research question (from our individual literature reviews), the models, the seeds
  and the metrics, and interpreted the results.

## Credits and licences
This repository is a fork of [ecg-fix](https://github.com/zackeberger/ecg-fix); all model code, benchmark and
pretrained weights are theirs. `ecg-fix` has no licence file at the root, so its code is used here only for
this course project. Third-party code in `external_src/` keeps its own licence: HeartLang (MIT), D-BETA
(CC BY-NC 4.0), CLOCS (CC BY-NC-SA 4.0), PTB-XL benchmarking code (GPL-3.0). Our additions are `analyze.py`,
`run_*_seeds.sh`, `analysis/` and `results/`.

Please cite the original work:

    @misc{berger2026ECGfix,
      title={Position: Evaluation of ECG Representations Must Be Fixed},
      author={Zachary Berger and Daniel Prakah-Asante and John Guttag and Collin M. Stultz},
      year={2026}, eprint={2602.17531}, archivePrefix={arXiv}, primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.17531}
    }

Data: PTB-XL 1.0.3, Wagner et al., *PTB-XL, a large publicly available electrocardiography dataset*,
Scientific Data, 2020. Models: MERL (Liu et al., ICML 2024) and CLOCS (Kiyasseh et al., ICML 2021).

Stockholm University.
