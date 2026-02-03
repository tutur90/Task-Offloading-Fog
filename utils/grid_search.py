

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