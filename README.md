# ETCA

Implementation of **Event-triggered continual adaptation of wearable recognition under resource budgets.**

## Install

Use Python 3.11 in an isolated environment. From the repository root:

```bash
python -m pip install -e .
```

## Data

The paper uses PAMAP2, MHEALTH, WISDM and Opportunity from [UCI](https://archive.ics.uci.edu/) and their dataset maintainers. Export resampled 2.56-second windows with 50% overlap as `x: (N,C,T)`, `y`, `subject`, `time`, `split`. Whole-subject roles are `train,calfit,caltemp,test` and optional `probe`. Preserve within-subject order. Resource fields define the charged cost contract; replace them with target measurements for physical resource studies. Reference energy/labels/latency are totals of the always-update calibration stream, rather than a single action.

## Workflow

```bash
python -m etca prepare --input data/wearable.npz
python -m etca train
python -m etca calibrate
python -m etca run
python -m etca evaluate
python -m etca plot
```

Main commands accept `--config config.json` and `--output runs/default`. Use a separate output directory for each seed and condition. Its `settings.json` records the resolved configuration.

## Configuration

`label_batch`, `adapt_epochs` and `adapt_learning_rate` specify the common DER++ action. `memory_bytes` limits windows, labels and historical logits. `utility_horizon` is used only in offline paired replay. `trigger_c` and the held-out temperature fit the utility trigger. Energy/label/update budgets and `worst_energy_j` are pre-decision feasibility constraints. Common training fields `epochs`, `batch_size`, `learning_rate`, `width` and `seed` control optimization, model size and randomness where present. Inspect `config.json` for defaults.

## Outputs

`utilities.json` records offline action targets. `trigger.npz` stores standardized logistic coefficients and temperature. `predictions.csv` contains pre-update predictions, decisions, labels acquired, resource charges and separate utility-audit labels. `metrics.csv` contains online AUC, online/final balanced accuracy, beneficial-update precision/recall, ECE, peak host RSS and resources. A separate `probe` subject partition enables `retention.csv` and average forgetting; its labels are evaluation-only.

## Method and comparisons

The learner uses four temporal CNN blocks, an LSTM, a bottleneck adapter and a classifier. The encoder is frozen online. Utility state uses representation drift, entropy, weak-view disagreement, sensor quality and remaining resources. Paired replay changes one bounded update and holds subsequent parameters fixed over its evaluation horizon. Deployment decisions are frozen before current/future labels are accessed by the audit.

Controls: `etca`, `never`, `always`, `periodic`, `entropy`, `adwin`, `no-pacing`, `no-temperature`. The run command evaluates these controls together. Changes to trainable objectives or architectures require a separate training run.

The `etca.compare` command averages repeated rows within independent units, then reports means, bootstrap intervals, paired Wilcoxon comparisons, rank-biserial effects and Holm adjustment per metric. For example:

```bash
python -m etca.compare --inputs "runs/*/metrics.csv" --reference etca --metrics online_auc acquired_labels charged_energy_j --unit seed
```

