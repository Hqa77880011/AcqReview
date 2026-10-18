# AcqReview

Implementation of acquisition-policy-aware early warning from sparse ICU measurements.

## Installation

Use Python 3.10+. Create and activate a virtual environment before installing:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

In PowerShell, replace the activation line with `.venv\Scripts\Activate.ps1`.

## Data and preprocessing

Obtain [MIMIC-IV](https://physionet.org/content/mimiciv/) or [eICU-CRD](https://physionet.org/content/eicu-crd/) under their source access agreements. Export `episodes.csv` with `patient,episode,split,end_hour,fatal,terminal_known,followup_hour`; hours are relative to ICU admission. `fatal` and `terminal_known` are 0/1. Use patient-disjoint `train,validation,calibration,test` roles (60/15/10/15%). Fatal termination must be confirmed within an active ICU segment using admission/death/transfers records; for eICU use unit discharge status and offsets. Ambiguous terminal events must be excluded before export. `observations.csv` contains `episode,hour,available_hour,variable,value`. Supply the later storage/result-availability time in `available_hour` and harmonize units before export. The exact 14 variable names are listed in `acqreview/__main__.py` as `VARIABLES`.

Preparation constructs only eligible hourly landmarks after six hours, stops before episode termination, and uses a 24-hour fatal-termination target. Confirmed nonfatal termination is negative; unknown outcomes require observed 24-hour follow-up. Observation thinning runs causally across each complete episode before prefixes are cut. Values, masks, ages, never-observed flags and group/count features are rebuilt for all seven policies. Training moments come from training episodes; padded bins do not update either GRU. Splits are checked by patient.

```bash
python -m acqreview prepare --episodes episodes.csv --observations observations.csv --config config.json
```

## Training and inference

```bash
python -m acqreview train --config config.json --output outputs
```

## Evaluation

```bash
python -m acqreview evaluate --config config.json --output outputs
python -m acqreview.analysis --input outputs/acquisition_timeline.npz --output outputs/patient_analysis.json --draws 1000
```

## Outputs

`data/landmarks.npz` holds histories. `outputs/` contains `model.pt`, `training.json`, `metrics.json`, `acquisition_timeline.npz`, `envelope.png` and optional patient-analysis JSON.

Keep `config.json` with the run; it contains model/data settings and the seed where the CLI does not expose one. Use separate output directories for different seeds/conditions.

Source inspection: Python 3.13.5, numpy 2.1.3, pandas 2.2.3, scipy 1.15.3, scikit-learn 1.6.1, torch 2.7.1+cu118, matplotlib 3.10.0 (installed versions inspected). Checks covered Python syntax, configuration parsing and CLI arguments. No training, full evaluation or CI was run in this update.
