from tqdm import tqdm


def run_pairwise(samples, outputs_a, outputs_b, judge):
    results = []

    for sample, a, b in tqdm(
        zip(samples, outputs_a, outputs_b),
        total=len(samples)
    ):
        try:
            result = judge.evaluate(sample, a, b)
        except Exception as e:
            result = {
                "winner": "Tie",
                "error": str(e)
            }

        results.append(result)

    return results