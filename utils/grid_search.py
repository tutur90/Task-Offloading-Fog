
import numpy as np
import os
from itertools import product

try:
    from skopt import Optimizer
    from skopt.space import Categorical
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False


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

def generate_parameter_grid(param_specs):
    """
    Generate a grid of parameter combinations from specifications.

    Args:
        param_specs: dict mapping "section.param" to list of values
            e.g., {
                "model.d_model": [64, 128, 256],
                "model.n_layers": [2, 3, 4],
                "training.lr": [0.001, 0.01]
            }

    Returns:
        List of dicts, each containing one parameter combination
    """
    if not param_specs:
        return []

    param_names = list(param_specs.keys())
    param_values = [param_specs[name] for name in param_names]

    grid = []
    for combination in product(*param_values):
        params = dict(zip(param_names, combination))
        grid.append(params)

    return grid


def generate_random_samples(param_specs, n_samples, method="lhs"):
    """
    Generate random parameter combinations from specifications.

    Args:
        param_specs: dict mapping "section.param" to list of values
            e.g., {
                "model.d_model": [64, 128, 256],
                "model.n_layers": [2, 3, 4],
                "training.lr": [0.001, 0.01]
            }
        n_samples: Number of random samples to generate
        method: Sampling method - "random", "lhs" (Latin Hypercube), or "sobol"

    Returns:
        List of dicts, each containing one random parameter combination
    """
    if not param_specs:
        return []

    param_names = list(param_specs.keys())
    param_values = [param_specs[name] for name in param_names]
    n_dims = len(param_names)

    max_combinations = int(np.prod([len(v) for v in param_values]))
    n_samples = min(n_samples, max_combinations)

    if method == "random":
        # Pure random sampling
        samples = []
        seen = set()
        while len(samples) < n_samples:
            combination = tuple(np.random.choice(values) for values in param_values)
            if combination not in seen:
                seen.add(combination)
                params = dict(zip(param_names, combination))
                samples.append(params)
        return samples

    elif method == "lhs":
        # Latin Hypercube Sampling - better space coverage
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=n_dims, seed=42)
        unit_samples = sampler.random(n=n_samples)

        samples = []
        seen = set()
        for unit_sample in unit_samples:
            combination = []
            for val_list, u in zip(param_values, unit_sample):
                idx = int(u * len(val_list))
                idx = min(idx, len(val_list) - 1)  # Clamp to valid range
                combination.append(val_list[idx])
            combination = tuple(combination)
            if combination not in seen:
                seen.add(combination)
                params = dict(zip(param_names, combination))
                samples.append(params)
        return samples

    elif method == "sobol":
        # Sobol sequence - quasi-random low-discrepancy sequence
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=n_dims, seed=42)
        unit_samples = sampler.random(n=n_samples)

        samples = []
        seen = set()
        for unit_sample in unit_samples:
            combination = []
            for val_list, u in zip(param_values, unit_sample):
                idx = int(u * len(val_list))
                idx = min(idx, len(val_list) - 1)
                combination.append(val_list[idx])
            combination = tuple(combination)
            if combination not in seen:
                seen.add(combination)
                params = dict(zip(param_names, combination))
                samples.append(params)
        return samples

    else:
        raise ValueError(f"Unknown sampling method: {method}. Use 'random', 'lhs', or 'sobol'")

class BayesianOptimizer:
    """Wrapper for scikit-optimize Bayesian optimization."""

    def __init__(self, param_specs, n_initial_points=5, random_state=42):
        """
        Initialize Bayesian optimizer.

        Args:
            param_specs: dict mapping "section.param" to list of values
            n_initial_points: Number of random initial points before Bayesian optimization kicks in
            random_state: Random seed for reproducibility
        """
        if not SKOPT_AVAILABLE:
            raise ImportError(
                "scikit-optimize is required for Bayesian optimization. "
                "Install it with: pip install scikit-optimize"
            )

        self.param_names = list(param_specs.keys())
        self.param_values = [param_specs[name] for name in self.param_names]

        # Create search space with Categorical dimensions
        self.space = [Categorical(values) for values in self.param_values]

        self.optimizer = Optimizer(
            dimensions=self.space,
            n_initial_points=n_initial_points,
            random_state=random_state,
            acq_func="EI"  # Expected Improvement
        )

        self.history = []

    def ask(self):
        """Get next parameter combination to try."""
        suggestion = self.optimizer.ask()
        params = dict(zip(self.param_names, suggestion))
        return params

    def tell(self, params, score):
        """Report the result of a parameter combination."""
        # Convert params dict back to list in correct order
        x = [params[name] for name in self.param_names]
        self.optimizer.tell(x, score)
        self.history.append((params, score))

    def get_best(self):
        """Get the best parameter combination found so far."""
        if not self.history:
            return None, None
        best_idx = np.argmin([h[1] for h in self.history])
        return self.history[best_idx]


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