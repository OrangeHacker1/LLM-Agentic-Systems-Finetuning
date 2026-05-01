import json

with open("results/phase4/checkpoint1_vs_checkpoint2.json") as f:
    data = json.load(f)

total = sum(data.values())
win_rate = data["A"] / total
loss_rate = data["B"] / total

report = {
    "checkpoint1_win_rate": win_rate,
    "checkpoint2_win_rate": loss_rate,
    "forgetting_detected": loss_rate > win_rate
}

with open("results/phase4/forgetting_report.json", "w") as f:
    json.dump(report, f, indent=2)

print(report)