import argparse
import json
from pathlib import Path
import numpy as np
from .common import binary_metrics, bootstrap, patient_weights, ratio


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/acquisition_timeline.npz")
    parser.add_argument("--output", default="outputs/patient_analysis.json")
    parser.add_argument("--draws", type=int, default=1000)
    args = parser.parse_args()
    data = np.load(args.input)
    weights = patient_weights(data["patient"])

    def auroc(ix):
        return binary_metrics(data["y"][ix], data["p"][ix], weights[ix])["auroc"]

    def review(ix):
        return ratio(100 * data["reviews"][ix].sum(), len(ix))

    result = {
        "auroc_ci95": bootstrap(data["patient"], auroc, args.draws, 11),
        "r100_ci95": bootstrap(data["patient"], review, args.draws, 11),
    }
    count = []
    for episode in np.unique(data["episode"]):
        mask = data["episode"] == episode
        if not data["fatal"][mask].any():
            continue
        end = data["event_time"][mask][0]
        eligible = mask & (data["time"] >= end - 24) & (data["time"] < end)
        count.append(
            {
                "episode": episode,
                "detected": bool(data["alerts"][eligible].any()),
                "reviewed": bool(data["reviews"][eligible].any()),
            }
        )
    result["episode_outcomes"] = count
    Path(args.output).write_text(
        json.dumps(json_ready(result), indent=2, allow_nan=False)
    )


if __name__ == "__main__":
    main()
