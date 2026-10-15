# AcqReview

Implementation of acquisition-policy-aware early warning from sparse ICU measurements.

```bash
python -m pip install -e .
python -m acqreview prepare --episodes episodes.csv --observations observations.csv
python -m acqreview train
python -m acqreview evaluate
```

Obtain [MIMIC-IV](https://physionet.org/content/mimiciv/) or [eICU-CRD](https://physionet.org/content/eicu-crd/) under their source access agreements. Export `episodes.csv` with `patient,episode,split,end_hour,fatal,terminal_known,followup_hour`; hours are relative to ICU admission. `fatal` and `terminal_known` are 0/1. Use patient-disjoint `train,validation,calibration,test` roles (60/15/10/15%). Fatal termination must be confirmed within an active ICU segment using admission/death/transfers records; for eICU use unit discharge status and offsets. Ambiguous terminal events must be excluded before export. `observations.csv` contains `episode,hour,available_hour,variable,value`. Supply the later storage/result-availability time in `available_hour` and harmonize units before export. The exact 14 variable names are listed in `acqreview/__main__.py` as `VARIABLES`.

Preparation constructs only eligible hourly landmarks after six hours, stops before episode termination, and uses a 24-hour fatal-termination target. Confirmed nonfatal termination is negative; unknown outcomes require observed 24-hour follow-up. Observation thinning runs causally across each complete episode before prefixes are cut. Values, masks, ages, never-observed flags and group/count features are rebuilt for all seven policies. Training moments come from training episodes; padded bins do not update either GRU. Splits are checked by patient.

Training fits the paper's 64-unit value GRU, 32-unit acquisition GRU and 32-unit fusion head with equal patient weighting. Calibration fits a positive temperature and chooses the largest original-alert threshold reaching the configured event sensitivity. Review uses upper-envelope threshold excursion and a positive calibration cutoff after six-hour suppression; threshold-proximity is a matched frozen-score control. Original high-score decisions retain alert priority.

`config.json` controls seed, optimizer, epochs, patience, event-sensitivity target and review workload. `--value-only` retrains the value branch. To compare policy banks, prepare a separate NPZ with `--bank point-drop` or `--bank no-lab`, keeping the same original model and data split. `--stress labs12`, `--stress bursts` and `--extra-delay 1` implement acquisition thinning, held-out charting interruptions and delayed availability. Retain the source model/calibration for frozen transfer by exporting external landmarks as a separate evaluation split and reusing source calibration records.

Outputs are `model.pt`, validation history, `metrics.json`, per-landmark timelines and `envelope.png`. Metrics cover patient-weighted AUROC/AUPRC, Brier, ten-bin ECE, calibration slope/intercept, event sensitivity, false alerts and review requests per 100 evaluable patient-hours, missed-episode review coverage, review precision, lead time and stable-decision errors. Each eligible hourly landmark contributes one hour of exposure. MR is null when no original warning misses an evaluable fatal episode. The reporting layer never alters the original probability or alert threshold. Other predictor architectures can supply frozen logits for the same policy histories; the included matched review controls use the fused scorer.

## Patient/cell analysis

Run `python -m acqreview.analysis --draws 1000` after evaluation to compute grouped bootstrap intervals and the supported subgroup or repeatability summaries. The input/output paths can be set with `--input` and `--output`.
