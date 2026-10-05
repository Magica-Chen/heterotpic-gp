"""Output locations and JSON serialization for the experiments."""
import json
import os
from pathlib import Path
import numpy as np
PACKAGE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('GP_OUTPUT_DIR', str(PACKAGE))).resolve()

def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value

def canonical(value):
    return json.dumps(clean_json(value), sort_keys=True, separators=(',', ':'), allow_nan=False)
