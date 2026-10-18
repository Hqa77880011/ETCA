# ETCA

Implementation of **Event-triggered continual adaptation of wearable recognition under resource budgets.**

## Installation

Requires Python >=3.10.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5, NumPy 2.1.3 and SciPy 1.15.3. Verification covers Python syntax, JSON/TOML and README CLI options against parser definitions; training and evaluation were not run.

## Data and preprocessing

The paper uses PAMAP2, MHEALTH, WISDM and Opportunity from [UCI](https://archive.ics.uci.edu/) and their dataset maintainers. Export resampled 2.56-second windows with 50% overlap as `x: (N,C,T)`, `y`, `subject`, `time`, `split`. Whole-subject roles are `train,calfit,caltemp,test` and optional `probe`. Preserve within-subject order. Resource fields define the charged cost contract; replace them with target measurements for physical resource studies. Reference energy/labels/latency are totals of the always-update calibration stream, rather than a single action.

```bash
python -m etca prepare --input data/wearable.npz --config config.json --output runs/full
```

## Training and inference

```bash
python -m etca train --config config.json --output runs/full
python -m etca calibrate --config config.json --output runs/full
python -m etca run --config config.json --output runs/full
```

`config.json` records the method settings and seed `13`; the resolved configuration is saved as `runs/full/settings.json`. Keep separate output directories for different seeds and variants.

## Evaluation

```bash
python -m etca evaluate --config config.json --output runs/full
```

## Ablations

Supported controls: `etca`, `never`, `always`, `periodic`, `entropy`, `adwin`, `no-pacing`, `no-temperature`.

The run command evaluates these controls together.

## Outputs

All main outputs are written below `runs/full/`.

`utilities.json` records offline action targets. `trigger.npz` stores standardized logistic coefficients and temperature. `predictions.csv` contains pre-update predictions, decisions, labels acquired, resource charges and separate utility-audit labels. `metrics.csv` contains online AUC, online/final balanced accuracy, beneficial-update precision/recall, ECE, peak host RSS and resources. A separate `probe` subject partition enables `retention.csv` and average forgetting; its labels are evaluation-only.

```bash
python -m etca plot --config config.json --output runs/full
```
