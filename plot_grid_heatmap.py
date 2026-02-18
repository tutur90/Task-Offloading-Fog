"""
Plot heatmaps from a grid search JSON file (e.g. grid_ga.json).

Usage:
    python plot_grid_heatmap.py <json_file> [--metric <0-3>] [--max <value>] [--out <path>]

Metrics index:
    0 - Task Success Rate (higher is better)
    1 - Avg Latency        (lower is better)
    2 - Avg Energy         (lower is better)
    3 - Score              (lower is better, default)
"""

import argparse
import json
import re
import sys

METRIC_NAMES = ["Task Success Rate", "Avg Latency", "Avg Energy", "Score"]


def parse_key(key: str) -> dict:
    """Parse 'section.param=val|section.param=val' into {param_name: value}."""
    result = {}
    for part in key.split("|"):
        m = re.match(r"^(.+)=(.+)$", part)
        if not m:
            continue
        name, raw_val = m.group(1), m.group(2)
        try:
            val = int(raw_val)
        except ValueError:
            try:
                val = float(raw_val)
            except ValueError:
                val = raw_val
        result[name] = val
    return result


def infer_param_specs(progress: dict) -> dict:
    """Infer parameter names and sorted unique values from all keys."""
    all_keys = list(progress.get("val_metrics", progress.get("test_metrics", {})).keys())
    param_specs: dict[str, set] = {}
    for key in all_keys:
        parsed = parse_key(key)
        for name, val in parsed.items():
            param_specs.setdefault(name, set()).add(val)
    return {name: sorted(vals) for name, vals in param_specs.items()}


def main():
    parser = argparse.ArgumentParser(description="Plot heatmap from grid search JSON")
    parser.add_argument("json_file", help="Path to the grid search JSON (e.g. grid_ga.json)")
    parser.add_argument("--metric", type=int, default=3, choices=[0, 1, 2, 3],
                        help="Metric index: 0=success_rate, 1=latency, 2=energy, 3=score (default: 3)")
    parser.add_argument("--max", type=float, default=None,
                        help="Max value for the colorbar scale. Values above are shown in black.")
    parser.add_argument("--out", default=None,
                        help="Output file path. Defaults to <json_file>_metric<N>.png next to the input file.")
    args = parser.parse_args()

    with open(args.json_file) as f:
        progress = json.load(f)

    if args.out is None:
        import os
        base = os.path.splitext(args.json_file)[0]
        args.out = f"{base}_metric{args.metric}.png"

    param_specs = infer_param_specs(progress)
    if len(param_specs) != 2:
        print(f"ERROR: Expected exactly 2 grid parameters, found {len(param_specs)}: {list(param_specs)}")
        sys.exit(1)

    from utils.plots import plot_grid_search_heatmap
    plot_grid_search_heatmap(
        param_specs=param_specs,
        progress=progress,
        metric_idx=args.metric,
        metric_name=METRIC_NAMES[args.metric],
        output_path=args.out,
        max_value=args.max,
    )



if __name__ == "__main__":
    main()
