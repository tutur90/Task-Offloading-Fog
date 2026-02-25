
import numpy as np
import os


def load_grid_search_progress(results_file):
    """Load previous grid search progress from file."""
    import json
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            data = json.load(f)
        return data
    return {"completed": {}, "val_metrics": {}, "test_metrics": {}}

def save_grid_search_progress(results_file, progress):
    """Save grid search progress to file."""
    import json
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(progress, f, indent=2)

def lambda_to_key(lambda_):
    """Convert lambda tuple to a string key for dict storage."""
    return f"{lambda_[0]:.6f},{lambda_[1]:.6f},{lambda_[2]:.6f}"

def params_to_key(params_dict):
    """Convert a parameter dict to a string key for dict storage."""
    sorted_items = sorted(params_dict.items())
    parts = []
    for k, v in sorted_items:
        if isinstance(v, (list, tuple)):
            v_str = ",".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in v)
            parts.append(f"{k}=[{v_str}]")
        elif isinstance(v, float):
            parts.append(f"{k}={v:.6f}")
        else:
            parts.append(f"{k}={v}")
    return "|".join(parts)

def generate_probability_grid(n_steps=11):
    """Generate grid of [l0, l1, l2] where l0 + l1 + l2 = 1"""
    grid = []
    step = 1.0 / (n_steps - 1)

    for i in range(n_steps):
        l0 = i * step
        for j in range(n_steps - i):
            l1 = j * step
            l2 = 1.0 - l0 - l1
            if l2 >= -1e-10:  # Handle floating point precision
                grid.append([l0, l1, max(0, l2)])

    return np.array(grid)

def apply_params_to_config(config, params):
    """
    Apply a parameter dict to a config, supporting nested keys with dot notation.

    Args:
        config: The config dict to modify (modified in place)
        params: Dict mapping "section.param" to values
            e.g., {"model.d_model": 256, "training.lr": 0.01}
    """
    for key, value in params.items():
        parts = key.split(".")
        target = config
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value
    return config

def parse_grid_search_params(param_strings):
    """
    Parse grid search parameters from command line strings.

    Args:
        param_strings: List of strings in format "section.param=val1,val2,val3"
            e.g., ["model.d_model=64,128,256", "model.n_layers=2,3,4"]

    Returns:
        Dict mapping param names to lists of values
    """
    param_specs = {}
    for param_str in param_strings:
        print(f"Parsing grid search param: {param_str}")
        key, values_str = param_str.split("=")
        
        values = []
        for v in values_str.split(","):
            v = v.strip()
            # Try to parse as int, then float, then keep as string
            try:
                values.append(int(v))
            except ValueError:
                try:
                    values.append(float(v))
                except ValueError:
                    values.append(v)
        param_specs[key] = values
    return param_specs
