import json
from collections import Counter

def safe_parse(x):
    try:
        return json.loads(x), None
    except Exception as e:
        return None, str(e)

def schema_ok(obj, schema):
    for k, t in schema.items():
        if k not in obj:
            return False
        if t == 'str' and not isinstance(obj[k], str):
            return False
        if t == 'int' and not isinstance(obj[k], int):
            return False
    return True

def compute_json_metrics(samples, preds):
    valid = 0
    compliant = 0
    exact = 0
    errors = Counter()
    for s, p in zip(samples, preds):
        obj, err = safe_parse(p)
        if obj is None:
            errors['invalid_json'] += 1
            continue
        valid += 1
        if schema_ok(obj, s['schema']):
            compliant += 1
        gold = json.loads(s['output'])
        if obj == gold:
            exact += 1
    n = len(samples)
    return {
        'json_validity': valid / n,
        'schema_compliance': compliant / n,
        'exact_match': exact / n,
        'errors': dict(errors)
    }