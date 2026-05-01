# Used in training part 2.
# utils/json_validator.py

import json


def is_valid_json(text):
    try:
        json.loads(text)
        return True
    except:
        return False
    