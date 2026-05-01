import json
import os

def save_all_outputs(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Save raw predictions
    with open(f"{output_dir}/json_results.json", "w") as f:
        json.dump(results["json"], f, indent=2)

    with open(f"{output_dir}/alpaca_results.json", "w") as f:
        json.dump(results["alpaca"], f, indent=2)

    # Metrics
    metrics = {
        "json_local_valid_rate": sum(r["local_valid"] for r in results["json"]) / max(len(results["json"]), 1),
        "json_teacher_pass_rate": sum(r["teacher_pass"] for r in results["json"]) / max(len(results["json"]), 1),
        "alpaca_avg_score": sum(r["score"] for r in results["alpaca"]) / max(len(results["alpaca"]), 1),
    }

    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved metrics:", metrics)

"""
def save_metrics(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    metrics = {
        "json_pass_rate": sum(r["json_pass"] for r in results["json"]) / max(len(results["json"]), 1),
        "alpaca_avg_score": sum(r["score"] for r in results["alpaca"]) / max(len(results["alpaca"]), 1),
    }

    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved metrics:", metrics)
"""

def save_metrics(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    json_results = results["json"]
    alpaca_results = results["alpaca"]

    metrics = {
        # JSON metrics
        "json_local_valid_rate": sum(r["local_valid"] for r in json_results) / max(len(json_results), 1),
        "json_teacher_pass_rate": sum(r["teacher_pass"] for r in json_results) / max(len(json_results), 1),

        # Alpaca metric
        "alpaca_avg_score": sum(r["score"] for r in alpaca_results) / max(len(alpaca_results), 1),
    }

    # Save metrics
    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save raw outputs (REQUIRED for assignment)
    with open(f"{output_dir}/json_results.json", "w") as f:
        json.dump(json_results, f, indent=2)

    with open(f"{output_dir}/alpaca_results.json", "w") as f:
        json.dump(alpaca_results, f, indent=2)

    print("Saved metrics:", metrics)


"""
import json
import os

def save_all_outputs(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Save raw predictions
    with open(f"{output_dir}/json_results.json", "w") as f:
        json.dump(results["json"], f, indent=2)

    with open(f"{output_dir}/alpaca_results.json", "w") as f:
        json.dump(results["alpaca"], f, indent=2)

    # Metrics
    metrics = {
        "json_local_valid_rate": sum(r["local_valid"] for r in results["json"]) / max(len(results["json"]), 1),
        "json_teacher_pass_rate": sum(r["teacher_pass"] for r in results["json"]) / max(len(results["json"]), 1),
        "alpaca_avg_score": sum(r["score"] for r in results["alpaca"]) / max(len(results["alpaca"]), 1),
    }

    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved metrics:", metrics)

""  "
def save_metrics(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    metrics = {
        "json_pass_rate": sum(r["json_pass"] for r in results["json"]) / max(len(results["json"]), 1),
        "alpaca_avg_score": sum(r["score"] for r in results["alpaca"]) / max(len(results["alpaca"]), 1),
    }

    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved metrics:", metrics)
""  "

def save_metrics(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    json_results = results["json"]
    alpaca_results = results["alpaca"]

    metrics = {
        # JSON metrics
        "json_local_valid_rate": sum(r["local_valid"] for r in json_results) / max(len(json_results), 1),
        "json_teacher_pass_rate": sum(r["teacher_pass"] for r in json_results) / max(len(json_results), 1),

        # Alpaca metric
        "alpaca_avg_score": sum(r["score"] for r in alpaca_results) / max(len(alpaca_results), 1),
    }

    # Save metrics
    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save raw outputs (REQUIRED for assignment)
    with open(f"{output_dir}/json_results.json", "w") as f:
        json.dump(json_results, f, indent=2)

    with open(f"{output_dir}/alpaca_results.json", "w") as f:
        json.dump(alpaca_results, f, indent=2)

    print("Saved metrics:", metrics)"""