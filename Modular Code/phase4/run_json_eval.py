import json
from metrics.json_metrics import compute_json_metrics

FILES = [
    "checkpoint0",
    "checkpoint1",
    "checkpoint2"
]

for ckpt in FILES:
    path = f"results/phase4/json_{ckpt}.json"

    with open(path) as f:
        data = json.load(f)

    samples = []
    preds = []

    for row in data:
        samples.append({
            "output": row["gold"],
            "schema": {}
        })
        preds.append(row["prediction"])

    metrics = compute_json_metrics(samples, preds)

    out = f"results/phase4/json_metrics_{ckpt}.json"
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)

    print(ckpt, metrics)