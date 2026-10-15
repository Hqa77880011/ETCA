import argparse
import copy
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
from scipy.integrate import trapezoid
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F


class Learner(nn.Module):
    def __init__(self, channels, c):
        super().__init__()
        layers = []
        for i in range(4):
            layers += [nn.Conv1d(channels if i == 0 else 64, 64, 5, padding=2), nn.ReLU()]
        self.conv = nn.Sequential(*layers)
        self.recurrent = nn.LSTM(64, 128, batch_first=True)
        self.adapter = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 128))
        self.head = nn.Linear(128, c['classes'])

    def forward(self, x):
        _, (hidden, _) = self.recurrent(self.conv(x).transpose(1, 2))
        z = hidden[-1]
        return self.head(z + self.adapter(z)), z


def prepare(c, out, source):
    d = dict(np.load(source, allow_pickle=False))
    if not {'x', 'y', 'subject', 'time', 'split'} <= set(d):
        raise ValueError('NPZ requires x(N,C,T),y,subject,time,split')
    for subject in np.unique(d['subject']):
        if len(np.unique(d['split'][d['subject'] == subject])) > 1:
            raise ValueError('A subject crosses train/calfit/caltemp/test/probe roles')
    fit = d['split'] == 'train'
    mean = d['x'][fit].mean((0, 2), keepdims=True)
    scale = d['x'][fit].std((0, 2), keepdims=True).clip(1e-6)
    d['x'] = np.nan_to_num((d['x'] - mean) / scale).astype(np.float32)
    np.savez_compressed(out / 'data.npz', **d)
    np.savez(out / 'normalization.npz', mean=mean, scale=scale)


def train(c, out):
    torch.manual_seed(c['seed'])
    d = np.load(out / 'data.npz')
    x, y = torch.tensor(d['x']), torch.tensor(d['y'], dtype=torch.long)
    model = Learner(x.shape[1], c)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['learning_rate'])
    ids = np.flatnonzero(d['split'] == 'train')
    for _ in range(c['epochs']):
        for selected in torch.tensor(ids)[torch.randperm(len(ids))].split(c['batch_size']):
            loss = F.cross_entropy(model(x[selected])[0], y[selected])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), out / 'base.pt')


def initialize(c, d, out):
    x, y = torch.tensor(d['x']), torch.tensor(d['y'], dtype=torch.long)
    model = Learner(x.shape[1], c)
    model.load_state_dict(torch.load(out / 'base.pt', weights_only=True, map_location='cpu'))
    model.eval()
    for module in [model.conv, model.recurrent]:
        for p in module.parameters():
            p.requires_grad_(False)
    ids = np.flatnonzero(d['split'] == 'train')
    per_sample = x[0].numel() * x.element_size() + c['classes'] * 4 + 8
    capacity = max(1, c['memory_bytes'] // per_sample)
    rng = np.random.default_rng(c['seed'])
    chosen = rng.choice(ids, min(capacity, len(ids)), replace=False)
    with torch.no_grad():
        logits, feature = model(x[chosen])
    memory = dict(x=x[chosen].clone(), y=y[chosen].clone(), logits=logits.clone(), seen=len(ids), capacity=capacity)
    return model, memory, feature.mean(0), feature.var(0).clamp_min(1e-5), x, y


def update(model, memory, x, y, c):
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=c['adapt_learning_rate'], weight_decay=1e-4)
    for _ in range(c['adapt_epochs']):
        output = model(x)[0]
        selected = torch.randperm(len(memory['y']))[:c['batch_size']]
        replay = model(memory['x'][selected])[0]
        loss = F.cross_entropy(output, y) + .5 * F.cross_entropy(replay, memory['y'][selected])
        loss += .5 * F.mse_loss(replay, memory['logits'][selected])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()


def insert(model, memory, x, y):
    with torch.no_grad():
        logits = model(x)[0]
    for value, label, logit in zip(x, y, logits):
        memory['seen'] += 1
        index = int(torch.randint(memory['seen'], (1,)))
        if len(memory['y']) < memory['capacity']:
            memory['x'] = torch.cat([memory['x'], value[None]])
            memory['y'] = torch.cat([memory['y'], label[None]])
            memory['logits'] = torch.cat([memory['logits'], logit[None]])
        elif index < memory['capacity']:
            memory['x'][index], memory['y'][index], memory['logits'][index] = value, label, logit


def state(model, x, reference, variance, fraction, remaining):
    with torch.no_grad():
        logits, feature = model(x)
        p = logits.softmax(-1)
        other = model(x + torch.randn_like(x) * .01)[0].softmax(-1)
        drift = ((feature.mean(0) - reference).square() / variance).mean()
        entropy = -(p * p.clamp_min(1e-8).log()).sum(1).mean()
        disagreement = (p - other).square().mean()
        quality = [float((x == 0).float().mean()), float((x.abs() > 4).float().mean()), float(x.var(unbiased=False))]
    return np.array([float(drift), float(entropy), float(disagreement), *quality, fraction, *remaining])


def utility(model, memory, current_x, current_y, future_x, future_y, c):
    candidate = copy.deepcopy(model)
    with torch.no_grad():
        old_future = (model(future_x)[0].argmax(1) == future_y).float().mean()
        old_retained = (model(memory['x'])[0].argmax(1) == memory['y']).float().mean()
    update(candidate, memory, current_x, current_y, c)
    with torch.no_grad():
        new_future = (candidate(future_x)[0].argmax(1) == future_y).float().mean()
        new_retained = (candidate(memory['x'])[0].argmax(1) == memory['y']).float().mean()
    recovery = float(new_future - old_future)
    forgetting = max(0., float(old_retained - new_retained))
    return recovery - .5 * forgetting - .2 * c['update_energy_j'] / c['reference_energy_j'] - .2 * len(current_y) / c['reference_labels'] - .1 * c['update_latency_ms'] / c['reference_latency_ms']


def calibrate(c, out):
    d = np.load(out / 'data.npz')
    records = []
    for role in ['calfit', 'caltemp']:
        model, memory, reference, variance, x, y = initialize(c, d, out)
        ids = np.flatnonzero(d['split'] == role)
        ids = ids[np.lexsort((d['time'][ids], d['subject'][ids]))]
        batches = list(np.array_split(ids, max(1, int(np.ceil(len(ids) / c['label_batch'])))))
        for index, batch in enumerate(batches[:-1]):
            remaining = [max(0, 1 - index / max(1, len(batches)))] * 3
            z = state(model, x[batch], reference, variance, index / len(batches), remaining)
            future = np.concatenate(batches[index + 1:index + 1 + c['utility_horizon']])
            value = utility(model, memory, x[batch], y[batch], x[future], y[future], c)
            records.append(dict(role=role, state=z.tolist(), utility=value, beneficial=value > 0))
            update(model, memory, x[batch], y[batch], c)
            insert(model, memory, x[batch], y[batch])
    train_records = [r for r in records if r['role'] == 'calfit']
    temp_records = [r for r in records if r['role'] == 'caltemp']
    features = np.array([r['state'] for r in train_records])
    scaler = StandardScaler().fit(features)
    classifier = LogisticRegression(C=c['trigger_c'], max_iter=2000).fit(scaler.transform(features), [r['beneficial'] for r in train_records])
    logits = classifier.decision_function(scaler.transform([r['state'] for r in temp_records]))
    labels = np.array([r['beneficial'] for r in temp_records])
    temperature = minimize_scalar(lambda logt: log_loss(labels, expit(logits / np.exp(logt)), labels=[0, 1]), bounds=(-4, 4), method='bounded')
    np.savez(out / 'trigger.npz', mean=scaler.mean_, scale=scaler.scale_, coefficient=classifier.coef_[0],
             intercept=classifier.intercept_[0], temperature=np.exp(temperature.x))
    (out / 'utilities.json').write_text(json.dumps(records))


def run(c, out):
    from river.drift import ADWIN
    d = np.load(out / 'data.npz')
    trigger = np.load(out / 'trigger.npz')
    ids = np.flatnonzero(d['split'] == 'test')
    ids = ids[np.lexsort((d['time'][ids], d['subject'][ids]))]
    batches = list(np.array_split(ids, max(1, int(np.ceil(len(ids) / c['label_batch'])))))
    records = []
    final_rows, retention = [], []
    for method in ['etca', 'never', 'always', 'periodic', 'entropy', 'adwin', 'no-pacing', 'no-temperature']:
        torch.manual_seed(c['seed'])
        model, memory, reference, variance, x, y = initialize(c, d, out)
        detector = ADWIN()
        probe_ids = np.flatnonzero(d['split'] == 'probe')
        peak_rss = 0.
        host_update_ms = 0.
        if len(probe_ids):
            with torch.no_grad():
                initial_probe = model(x[probe_ids])[0].argmax(1)
            for label in np.unique(d['y'][probe_ids]):
                selected = y[probe_ids] == int(label)
                retention.append(dict(method=method,event=-1,label=int(label),accuracy=float((initial_probe[selected] == y[probe_ids][selected]).float().mean())))
        used_energy, labels, updates, last = 0., 0, 0, -c['refractory_batches']
        for index, batch in enumerate(batches):
            remaining = [1 - used_energy / c['energy_budget_j'], 1 - labels / c['label_budget'], 1 - updates / c['update_budget']]
            z = state(model, x[batch], reference, variance, index / len(batches), remaining)
            temperature = 1 if method == 'no-temperature' else float(trigger['temperature'])
            probability = float(expit((((z - trigger['mean']) / trigger['scale']) @ trigger['coefficient'] + trigger['intercept']) / temperature))
            spent = max(1 - np.array(remaining))
            threshold = np.clip(.5 + .4 * (spent - index / len(batches)), .2, .9)
            if method == 'no-pacing':
                threshold = .5
            detector.update(z[1])
            proposed = probability >= threshold
            if method == 'never':
                proposed = False
            elif method == 'always':
                proposed = True
            elif method == 'periodic':
                proposed = index % 4 == 0
            elif method == 'entropy':
                proposed = z[1] >= c['entropy_threshold']
            elif method == 'adwin':
                proposed = detector.drift_detected
            feasible = (used_energy + c['worst_energy_j'] <= c['energy_budget_j'] and
                        labels + len(batch) <= c['label_budget'] and updates + 1 <= c['update_budget'])
            action = bool(proposed and feasible and (index - last >= c['refractory_batches'] if method.startswith('etca') else True))
            with torch.no_grad():
                predicted = model(x[batch])[0].argmax(1).numpy()
            beneficial = np.nan
            if index + 1 < len(batches):
                future = np.concatenate(batches[index + 1:index + 1 + c['utility_horizon']])
                saved_rng = torch.get_rng_state()
                beneficial = utility(model, memory, x[batch], y[batch], x[future], y[future], c) > 0
                torch.set_rng_state(saved_rng)
            if action:
                update_start = time.perf_counter()
                update(model, memory, x[batch], y[batch], c)
                insert(model, memory, x[batch], y[batch])
                used_energy += c['update_energy_j']
                labels += len(batch)
                updates += 1
                last = index
                host_update_ms += 1000 * (time.perf_counter() - update_start)
                if len(probe_ids):
                    with torch.no_grad():
                        probe_prediction = model(x[probe_ids])[0].argmax(1)
                    for label in np.unique(d['y'][probe_ids]):
                        selected = y[probe_ids] == int(label)
                        retention.append(dict(method=method,event=index,label=int(label),accuracy=float((probe_prediction[selected] == y[probe_ids][selected]).float().mean())))
            peak_rss = max(peak_rss, psutil.Process().memory_info().rss / 2 ** 20)
            for position, prediction in zip(batch, predicted):
                records.append(dict(method=method, event=index, position=int(position), subject=d['subject'][position],
                                    y=int(y[position]), prediction=int(prediction), probability=probability,
                                    beneficial=beneficial, action=action, updates=updates, labels=labels,
                                    charged_energy_j=used_energy, cumulative_latency_ms=updates * c['update_latency_ms']))
        with torch.no_grad():
            final_prediction = []
            for group in torch.tensor(ids).split(c['batch_size']):
                final_prediction.extend(model(x[group])[0].argmax(1).numpy())
        final_rows.append(dict(method=method, final_balanced_accuracy=balanced_accuracy_score(y[ids], final_prediction), peak_rss_mib=peak_rss, host_update_ms=host_update_ms))
        torch.save(model.state_dict(), out / f'final_{method}.pt')
    pd.DataFrame(records).to_csv(out / 'predictions.csv', index=False)
    pd.DataFrame(final_rows).to_csv(out / 'final.csv', index=False)
    if retention:
        pd.DataFrame(retention).to_csv(out / 'retention.csv', index=False)


def evaluate(c, out):
    d = pd.read_csv(out / 'predictions.csv')
    rows = []
    for method, g in d.groupby('method'):
        by_event = g.groupby('event').apply(lambda frame: (frame.y == frame.prediction).mean())
        events_frame = g.drop_duplicates('event')
        scored = events_frame[events_frame.beneficial.notna()]
        actual = scored.beneficial.astype(bool)
        confidence = scored.probability.to_numpy()
        ece = sum(len(ids) / len(scored) * abs(actual.iloc[ids].mean() - confidence[ids].mean()) for ids in np.array_split(np.argsort(confidence), min(15, len(scored)))) if len(scored) else np.nan
        rows.append(dict(method=method, online_auc=float(trapezoid(by_event.to_numpy(), dx=1 / max(1, len(by_event) - 1))) if len(by_event) > 1 else by_event.iloc[0],
                         online_balanced_accuracy=balanced_accuracy_score(g.y, g.prediction),
                         update_count=int(g.updates.max()), acquired_labels=int(g.labels.max()), charged_energy_j=g.charged_energy_j.max(),
                         cumulative_update_latency_ms=g.cumulative_latency_ms.max(),
                         trigger_precision=precision_score(actual, scored.action, zero_division=0),
                         trigger_recall=recall_score(actual, scored.action, zero_division=0), trigger_ece=ece))
    result = pd.DataFrame(rows).merge(pd.read_csv(out / 'final.csv'), on='method')
    if (out / 'retention.csv').exists():
        old = pd.read_csv(out / 'retention.csv')
        forgetting = []
        for method, group in old.groupby('method'):
            changes = []
            for _, curve in group.groupby('label'):
                changes.append(curve.accuracy.max() - curve.sort_values('event').accuracy.iloc[-1])
            forgetting.append(dict(method=method, average_forgetting=np.mean(changes)))
        result = result.merge(pd.DataFrame(forgetting), on='method')
    result.to_csv(out / 'metrics.csv', index=False)


def plot(out):
    d = pd.read_csv(out / 'predictions.csv')
    fig, ax = plt.subplots(figsize=(8, 4))
    for method, g in d.groupby('method'):
        curve = (g.y == g.prediction).expanding().mean()
        ax.plot(np.linspace(0, 1, len(g)), curve, label=method)
    ax.set(xlabel='Stream fraction', ylabel='Cumulative online accuracy')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'online.png', dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'train', 'calibrate', 'run', 'evaluate', 'plot'])
    p.add_argument('--input')
    p.add_argument('--config', default='config.json')
    p.add_argument('--output', default='runs/default')
    args = p.parse_args()
    c = json.loads(Path(args.config).read_text(encoding='utf-8'))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    settings = out / 'settings.json'
    settings.write_text(json.dumps(c, indent=2))
    if args.command == 'prepare':
        if not args.input:
            p.error('--input required')
        prepare(c, out, args.input)
    elif args.command == 'plot':
        plot(out)
    else:
        globals()[args.command](c, out)


if __name__ == '__main__':
    main()
