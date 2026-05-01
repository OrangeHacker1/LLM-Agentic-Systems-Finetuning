import json
from judge.llm_judge import LLMJudge
from judge.pairwise_eval import run_pairwise


def summarize(results):
    wins = sum(r["winner"] == "A" for r in results)
    losses = sum(r["winner"] == "B" for r in results)
    ties = sum(r["winner"] == "Tie" for r in results)
    n = len(results)

    return {
        "win_rate": wins / n,
        "loss_rate": losses / n,
        "tie_rate": ties / n
    }


pairs = [
    ("checkpoint0", "checkpoint1"),
    ("checkpoint1", "checkpoint2")
]

judge = LLMJudge()

for a, b in pairs:
    with open(f"results/phase4/alpaca_{a}.json") as f:
        A = json.load(f)

    with open(f"results/phase4/alpaca_{b}.json") as f:
        B = json.load(f)

    samples = [
        {
            "instruction": x["instruction"],
            "input": ""
        }
        for x in A
    ]

    outputs_a = [x["prediction"] for x in A]
    outputs_b = [x["prediction"] for x in B]

    results = run_pairwise(samples, outputs_a, outputs_b, judge)
    summary = summarize(results)

    with open(f"results/phase4/{a}_vs_{b}.json", "w") as f:
        json.dump({
            "summary": summary,
            "details": results
        }, f, indent=2)

    print(a, "vs", b, summary)