import json
from utils.io import write_json, read_json

summary = {}

files = [
    "results/phase4/pairwise_01.json",
    "results/phase4/pairwise_12.json",
    "results/phase4/forgetting.json"
]

for path in files:
    try:
        summary[path] = read_json(path)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Error reading {path}: {e}")

write_json("results/phase4/final_summary.json", summary)

print("Saved final_summary.json")