import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from .common import (
    binary_metrics,
    check_groups,
    fit,
    patient_weights,
    probabilities,
    ratio,
    temperature,
)


def json_ready(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (not np.isfinite(value)):
        return None
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(v) for v in value]
    return value


VARIABLES = [
    "heart_rate",
    "map",
    "respiratory_rate",
    "spo2",
    "temperature",
    "sodium",
    "potassium",
    "chloride",
    "bicarbonate",
    "bun",
    "creatinine",
    "glucose",
    "wbc",
    "platelets",
]
POLICIES = [(0, 0), (2, 0), (4, 0), (0, 6), (0, 12), (2, 6), (4, 12)]


def rebuild(raw, valid, means, scales):
    values = np.zeros_like(raw)
    age = np.full_like(raw, 48.0)
    never = np.ones_like(raw)
    counts = valid.sum(1)
    previous = means.copy()
    last = np.full(14, -np.inf)
    for h in range(len(raw)):
        observed = np.isfinite(raw[h]) & valid[h]
        previous[observed] = raw[h, observed]
        last[observed] = h
        values[h] = (previous - means) / scales
        age[h] = np.minimum(48, h - last)
        never[h] = ~np.isfinite(last)
    return np.concatenate(
        [
            values,
            valid.astype(float),
            age / 48,
            never,
            counts[:, None] / 14,
            valid[:, 5:12].any(1)[:, None],
            valid[:, 12:].any(1)[:, None],
        ],
        1,
    )


def thin(mask, vital_gap, lab_gap):
    keep = mask.copy()
    last = np.full(7, -np.inf)
    for h in range(len(mask)):
        for group, columns in enumerate(
            [[0], [1], [2], [3], [4], list(range(5, 12)), [12, 13]]
        ):
            gap = vital_gap if group < 5 else lab_gap
            available = mask[h, columns]
            if available.any():
                if h - last[group] < gap:
                    keep[h, columns] = False
                else:
                    last[group] = h
    return keep


def prepare(args, cfg):
    episodes = pd.read_csv(args.episodes, dtype={"patient": str, "episode": str})
    obs = pd.read_csv(args.observations, dtype={"episode": str})
    check_groups(episodes.patient.to_numpy(), episodes.split.to_numpy())
    if ((episodes.fatal == 1) & (episodes.terminal_known != 1)).any():
        raise ValueError("Fatal endpoints require verified terminal status")
    if not set(obs.variable).issubset(VARIABLES):
        raise ValueError("Unknown variable in observation dictionary")
    training = set(episodes.loc[episodes.split == "train", "episode"])
    means = np.array(
        [
            obs.loc[obs.episode.isin(training) & (obs.variable == v), "value"].mean()
            for v in VARIABLES
        ]
    )
    scales = np.array(
        [
            obs.loc[obs.episode.isin(training) & (obs.variable == v), "value"].std()
            for v in VARIABLES
        ]
    )
    if not np.isfinite(means).all():
        raise ValueError("Every variable needs training observations")
    scales = np.where(np.isfinite(scales) & (scales > 0), scales, 1.0)
    records = []
    for ep in episodes.itertuples(index=False):
        end = float(ep.end_hour)
        hours = int(np.ceil(end))
        raw = np.full((hours, 14), np.nan)
        visits = obs.loc[obs.episode == ep.episode].copy()
        visits["availability"] = (
            np.maximum(visits.hour, visits.available_hour) + args.extra_delay
        )
        visits = visits.sort_values(["availability", "hour"])
        for row in visits.itertuples(index=False):
            h = max(0, int(np.ceil(row.availability)) - 1)
            if h < hours:
                raw[h, VARIABLES.index(row.variable)] = row.value
        native = np.isfinite(raw)
        if args.stress == "bursts":
            native[np.isin(np.arange(hours) % 24, [6, 7, 8, 18, 19, 20])] = False
        if args.stress == "labs12":
            native = thin(native, 0, 12)
        transformed = []
        for bank, (vg, lg) in enumerate(POLICIES):
            keep = thin(native, vg, lg)
            if args.bank == "point-drop" and bank:
                rng = np.random.default_rng(cfg["seed"] + bank)
                keep = native & (rng.random(native.shape) >= 0.25)
            if args.bank == "no-lab" and bank >= 3:
                keep = native.copy()
            transformed.append(keep)
        for h in range(6, hours + 1):
            if h >= end:
                continue
            if not ep.terminal_known and ep.followup_hour < h + 24:
                continue
            y = int(bool(ep.fatal) and h < end <= h + 24)
            start = max(0, h - 24)
            length = h - start
            x = np.zeros((7, 24, 59), np.float32)
            for k in range(7):
                x[k, -length:] = rebuild(
                    raw[start:h], transformed[k][start:h], means, scales
                )
            padding = np.r_[np.zeros(24 - length), np.ones(length)]
            records.append(
                (x, padding, y, ep.patient, ep.episode, h, end, int(ep.fatal), ep.split)
            )
    out = Path(args.data)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        **{
            name: np.asarray([r[i] for r in records])
            for i, name in enumerate(
                [
                    "x",
                    "padding",
                    "y",
                    "patient",
                    "episode",
                    "time",
                    "event_time",
                    "fatal",
                    "split",
                ]
            )
        },
        means=means,
        scales=scales,
    )


class Predictor(nn.Module):

    def __init__(self, value_only=False):
        super().__init__()
        self.value_only = value_only
        self.values = nn.GRUCell(28, 64)
        self.acquisition = nn.GRUCell(45, 32)
        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(64 if value_only else 96, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x, padding):
        hv = x.new_zeros(len(x), 64)
        ha = x.new_zeros(len(x), 32)
        for t in range(x.shape[1]):
            active = padding[:, t, None]
            hv = active * self.values(x[:, t, :28], hv) + (1 - active) * hv
            acquisition = torch.cat([x[:, t, 14:28], x[:, t, 28:]], 1)
            ha = active * self.acquisition(acquisition, ha) + (1 - active) * ha
        return self.head(hv if self.value_only else torch.cat([hv, ha], 1)).squeeze(1)


def timeline(d, ix, p, upper, tau, eta, method="acquisition"):
    alerts = np.zeros(len(ix), bool)
    reviews = np.zeros(len(ix), bool)
    score = np.where(
        p < tau, np.maximum(0, upper - tau) if method == "acquisition" else p, 0.0
    )
    for episode in np.unique(d["episode"][ix]):
        positions = np.flatnonzero(d["episode"][ix] == episode)
        positions = positions[np.argsort(d["time"][ix][positions])]
        last_alert = last_review = -np.inf
        for j in positions:
            t = d["time"][ix[j]]
            if p[j] >= tau:
                if t - last_alert >= 6:
                    alerts[j] = True
                    last_alert = t
            elif score[j] > 0 and score[j] >= eta and (t - last_review >= 6):
                reviews[j] = True
                last_review = t
    fatal_episodes = [
        e
        for e in np.unique(d["episode"][ix])
        if d["fatal"][ix][d["episode"][ix] == e].any()
    ]
    detected = missed_review = 0
    lead = []
    for episode in fatal_episodes:
        mask = d["episode"][ix] == episode
        end = d["event_time"][ix][mask][0]
        eligible = mask & (d["time"][ix] >= end - 24) & (d["time"][ix] < end)
        found = alerts & eligible
        if found.any():
            detected += 1
            lead.append(end - np.min(d["time"][ix][found]))
        elif (reviews & eligible).any():
            missed_review += 1
    return (
        {
            "sensitivity": ratio(detected, len(fatal_episodes)),
            "missed_episode_proportion": ratio(
                len(fatal_episodes) - detected, len(fatal_episodes)
            ),
            "mr": ratio(missed_review, len(fatal_episodes) - detected),
            "fa100": 100 * float(np.sum(alerts & (d["y"][ix] == 0))) / len(ix),
            "r100": 100 * float(reviews.sum()) / len(ix),
            "review_precision": ratio(
                np.sum(reviews & (d["y"][ix] == 1)), reviews.sum()
            ),
            "warning_lead_h": float(np.median(lead)) if lead else None,
            "events": len(fatal_episodes),
        },
        alerts,
        reviews,
        score,
    )


def infer(model, d, ix):
    results = []
    with torch.no_grad():
        for bank in range(7):
            chunks = []
            for batch in np.array_split(ix, max(1, int(np.ceil(len(ix) / 128)))):
                chunks.append(
                    model(
                        torch.tensor(d["x"][batch, bank]),
                        torch.tensor(d["padding"][batch], dtype=torch.float32),
                    ).numpy()
                )
            results.append(np.concatenate(chunks))
    return np.stack(results, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "train", "evaluate"])
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--data", default="data/landmarks.npz")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--episodes")
    parser.add_argument("--observations")
    parser.add_argument("--extra-delay", type=float, default=0)
    parser.add_argument(
        "--stress", choices=["native", "bursts", "labs12"], default="native"
    )
    parser.add_argument(
        "--bank", choices=["grouped", "point-drop", "no-lab"], default="grouped"
    )
    parser.add_argument("--value-only", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    torch.manual_seed(cfg["seed"])
    if args.command == "prepare":
        prepare(args, cfg)
        return
    d = np.load(args.data)
    check_groups(d["patient"], d["split"])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.command == "train":
        model = Predictor(args.value_only)
        cfg["value_only"] = args.value_only

        def items(split):
            ix = np.flatnonzero(d["split"] == split)
            return tuple(
                (
                    torch.tensor(a, dtype=torch.float32)
                    for a in [
                        d["x"][ix, 0],
                        d["padding"][ix],
                        d["y"][ix],
                        patient_weights(d["patient"][ix]),
                    ]
                )
            )

        def objective(m, x, pad, y, w):
            return (
                (
                    F.binary_cross_entropy_with_logits(m(x, pad), y, reduction="none")
                    * w
                ).sum(),
                w.sum(),
            )

        fit(model, items("train"), items("validation"), objective, cfg, out)
        return
    checkpoint = torch.load(out / "model.pt", map_location="cpu", weights_only=False)
    model = Predictor(checkpoint["config"]["value_only"])
    model.load_state_dict(checkpoint["state"])
    model.eval()
    cal = np.flatnonzero(d["split"] == "calibration")
    test = np.flatnonzero(d["split"] == "test")
    if not len(cal) or not len(test):
        raise ValueError("Calibration and test patients are required")
    cal_logits = infer(model, d, cal)
    temp = temperature(
        cal_logits[:, 0], d["y"][cal], patient_weights(d["patient"][cal])
    )
    cp = probabilities(cal_logits, temp)
    thresholds = np.unique(np.r_[0.0, cp[:, 0], 1.0])
    feasible = [
        t
        for t in thresholds
        if (timeline(d, cal, cp[:, 0], cp.max(1), t, np.inf)[0]["sensitivity"] or 0)
        >= cfg["sensitivity_target"]
    ]
    if not feasible:
        raise ValueError("No warning threshold meets calibration sensitivity")
    tau = float(max(feasible))
    logits = infer(model, d, test)
    probs = probabilities(logits, temp)
    lower, upper = (probs.min(1), probs.max(1))
    p = probs[:, 0]
    w = patient_weights(d["patient"][test])
    rows = []
    for method in ["acquisition", "proximity"]:
        scores = np.where(
            cp[:, 0] < tau,
            np.maximum(0, cp.max(1) - tau) if method == "acquisition" else cp[:, 0],
            0.0,
        )
        candidates = np.unique(np.r_[scores[scores > 0], np.inf])
        eta = np.inf
        for candidate in candidates:
            m = timeline(d, cal, cp[:, 0], cp.max(1), tau, candidate, method)[0]
            if m["r100"] <= cfg["review_target_per_100h"]:
                eta = float(candidate)
                break
        metrics, alerts, reviews, score = timeline(d, test, p, upper, tau, eta, method)
        stable_high = lower >= tau
        stable_low = upper < tau
        metrics.update(
            method=method,
            tau=tau,
            eta=eta,
            **binary_metrics(d["y"][test], p, w),
            stable_coverage=float(np.average(stable_high | stable_low, weights=w)),
            stable_low_error=ratio(
                np.sum(w * stable_low * d["y"][test]), np.sum(w * stable_low)
            ),
            stable_high_error=ratio(
                np.sum(w * stable_high * (1 - d["y"][test])), np.sum(w * stable_high)
            ),
        )
        rows.append(metrics)
        np.savez(
            out / f"{method}_timeline.npz",
            patient=d["patient"][test],
            episode=d["episode"][test],
            time=d["time"][test],
            y=d["y"][test],
            p=p,
            lower=lower,
            upper=upper,
            alerts=alerts,
            reviews=reviews,
            score=score,
            event_time=d["event_time"][test],
            fatal=d["fatal"][test],
        )
    (out / "metrics.json").write_text(
        json.dumps(
            json_ready({"temperature": temp, "methods": rows}),
            indent=2,
            allow_nan=False,
        )
    )
    plt.hist(upper - lower, bins=30)
    plt.xlabel("Policy-envelope width")
    plt.ylabel("Landmarks")
    plt.tight_layout()
    plt.savefig(out / "envelope.png")
    plt.close()


if __name__ == "__main__":
    main()
