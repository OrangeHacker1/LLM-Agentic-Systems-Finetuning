import json
from datasets import load_dataset
from phase4.inference import load_checkpoint, generate_response

CHECKPOINTS = {
    "checkpoint0": "base",
    "checkpoint1": "stage1_adapter",
    "checkpoint2": "stage2_adapter"
}

EVAL_FILES = {
    "alpaca": "data/alpaca_eval.jsonl",
    "json": "data/teacher_json_eval.jsonl"
}


def run_eval(task_name, path):
    dataset = load_dataset("json", data_files=path)["train"]

    for ckpt_name, ckpt_path in CHECKPOINTS.items():
        model, tokenizer = load_checkpoint(ckpt_path)

        outputs = []

        for item in dataset:
            prompt = item["instruction"] + "\n" + item.get("input", "")
            pred = generate_response(model, tokenizer, prompt)

            outputs.append({
                "instruction": item["instruction"],
                "gold": item["output"],
                "prediction": pred
            })

        save_path = f"results/phase4/{task_name}_{ckpt_name}.json"
        with open(save_path, "w") as f:
            json.dump(outputs, f, indent=2)


if __name__ == "__main__":
    for task, path in EVAL_FILES.items():
        run_eval(task, path)