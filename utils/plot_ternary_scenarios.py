#!/usr/bin/env python3
"""
Plot ternary subplots for all scenarios and policies.

Layout: two figures per scenario (Val and Test), one subplot per policy.
The ternary grid resolution adapts automatically per scenario.
Colorbars are in log scale.

Usage:
    python utils/plot_ternary_scenarios.py [logs_dir] [--metric 0-3] [--max-value X] [--output-dir DIR] [--dpi N]

Metrics:
    0 = Success Rate
    1 = Latency (ms)
    2 = Energy (mJ)
    3 = Score (default)
"""

import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


METRIC_NAMES = ["Success Rate", "Latency (ms)", "Energy (mJ)", "Score"]


class _LogPowerTransform(mcolors.AsinhNorm.__class__):  # noqa: use FuncNorm pattern
    pass


class LogPowerNorm(mcolors.Normalize):
    """
    Two-stage norm: log-transform the data, then apply a power (gamma < 1).
    This gives far more colormap space to small values than LogNorm alone.

    Effective transform: ((log(v) - log(vmin)) / (log(vmax) - log(vmin))) ** gamma
    """

    def __init__(self, vmin, vmax, gamma=0.3, clip=False):
        super().__init__(vmin=vmin, vmax=vmax, clip=clip)
        self.gamma = gamma
        self._log_vmin = np.log(vmin)
        self._log_vmax = np.log(vmax)

    def __call__(self, value, clip=None):
        value = np.ma.asarray(value, dtype=float)
        # Clip to [vmin, vmax] range
        value = np.ma.clip(value, self.vmin, self.vmax)
        # Log-normalize to [0, 1], then apply power
        log_range = self._log_vmax - self._log_vmin
        normed = (np.log(value) - self._log_vmin) / log_range
        return np.ma.power(normed, self.gamma)

    def inverse(self, value):
        value = np.asarray(value)
        log_range = self._log_vmax - self._log_vmin
        # Reverse power, then reverse log-normalize
        return np.exp(np.power(value, 1.0 / self.gamma) * log_range + self._log_vmin)


def _logpower_colorbar_ticks(norm):
    """
    Returns (all_ticks, labeled_ticks):
    - all_ticks: ×1..×9 for each decade (tick bars)
    - labeled ticks: only ×1, ×2, ×5 per decade (written labels)
    """
    log_min = np.log10(norm.vmin)
    log_max = np.log10(norm.vmax)
    decades = np.arange(np.floor(log_min), np.ceil(log_max) + 1)
    all_mults    = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
    labeled_mults = np.array([1, 2, 5], dtype=float)

    all_ticks     = np.unique(np.concatenate([10.0 ** d * all_mults    for d in decades]))
    labeled_ticks = np.unique(np.concatenate([10.0 ** d * labeled_mults for d in decades]))

    in_range = lambda t: t[(t >= norm.vmin) & (t <= norm.vmax)]
    return in_range(all_ticks), in_range(labeled_ticks)


def _draw_ternary_on_ax(ax, grid, values, title, labels=None, cmap="viridis_r", s=20, norm=None):
    """Draw a ternary plot on an existing Axes object."""
    if labels is None:
        labels = ["λ0", "λ1", "λ2"]

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_over("black")

    # Convert to 2D ternary coordinates
    x = 0.5 * (2 * grid[:, 1] + grid[:, 2])
    y = (np.sqrt(3) / 2) * grid[:, 2]

    # Clip values to norm range so set_over fires correctly
    if norm is not None:
        plot_vals = np.clip(values, norm.vmin, norm.vmax * 1.001)
    else:
        plot_vals = values

    # Triangle boundary (drawn first, behind points)
    ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3) / 2, 0], "k-", linewidth=1.5, zorder=1)

    sc = ax.scatter(x, y, c=plot_vals, cmap=cmap_obj, s=s, norm=norm, zorder=2)

    # Vertex labels — tight to the triangle edges
    ax.text(0,  -0.02, labels[0], fontsize=10, fontweight="bold", ha="center", va="top")
    ax.text(1,  -0.02, labels[1], fontsize=10, fontweight="bold", ha="center", va="top")
    ax.text(0.5, np.sqrt(3) / 2 + 0.01, labels[2], fontsize=10, fontweight="bold", ha="center", va="bottom")

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=6)

    cb = plt.colorbar(sc, ax=ax, shrink=0.65, extend="max", pad=0.02)
    # Override ticks with original-scale decade values for readability
    if isinstance(norm, LogPowerNorm):
        all_ticks, labeled_ticks = _logpower_colorbar_ticks(norm)
        labeled_set = set(np.round(labeled_ticks, 10))
        cb.set_ticks(all_ticks)
        cb.ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v:g}" if round(v, 10) in labeled_set else "")
        )
        cb.ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    return sc


def load_grid_data(progress_path):
    with open(progress_path) as f:
        data = json.load(f)

    keys = list(data["val_metrics"].keys())
    grid = np.array([[float(v) for v in k.split(",")] for k in keys])
    val_metrics = np.array([data["val_metrics"][k] for k in keys])
    test_metrics = np.array([data["test_metrics"][k] for k in keys])
    return grid, val_metrics, test_metrics


def point_size_for_grid(n_pts):
    """Scale point size inversely with number of grid points."""
    return max(6, int(2000 / n_pts))


def make_norm(values_list, max_value=None, gamma=0.3):
    """Build a LogPowerNorm from the given value arrays."""
    all_vals = np.concatenate([v.ravel() for v in values_list])
    all_vals = all_vals[all_vals > 0]
    vmin = float(np.min(all_vals))
    vmax = max_value if max_value is not None else float(np.percentile(all_vals, 90))
    vmax = max(vmax, vmin * 1.01)
    return LogPowerNorm(vmin=vmin, vmax=vmax, gamma=gamma)


def save_figure(fig, scenario_label, split, metric_name, logs_dir, scenario_key, output_dir, dpi):
    safe_name = scenario_label.replace("/", "_").replace("\\", "_")
    split_tag = split.lower()
    fig_name = f"ternary_{safe_name}_{split_tag}_{metric_name.lower().replace(' ', '_')}.png"

    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = logs_dir / scenario_key

    out_path = out_dir / fig_name
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs_dir", nargs="?", default="logs", help="Root logs directory (default: logs)")
    parser.add_argument("--metric", type=int, default=3, choices=[0, 1, 2, 3],
                        help="Metric index: 0=success, 1=latency, 2=energy, 3=score (default)")
    parser.add_argument("--max-value", type=float, default=None,
                        help="Colorbar max; values above shown in black. Default: 90th percentile per scenario.")
    parser.add_argument("--gamma", type=float, default=0.3,
                        help="Power exponent applied after log transform (default: 0.3). "
                             "Lower = more colormap space for small values. Range: (0, 1].")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save figures. Default: alongside each scenario's data.")
    parser.add_argument("--dpi", type=int, default=200, help="Output DPI (default: 200)")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    if not logs_dir.exists():
        print(f"Error: logs directory '{logs_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    metric_name = METRIC_NAMES[args.metric]

    # Collect all lambda_grid_search_progress.json files, grouped by scenario
    # Expected layout: <logs_dir>/<dataset>/<subscenario>/<policy>/lambda_grid_search_progress.json
    scenarios: dict[Path, dict[str, Path]] = {}
    for pf in sorted(logs_dir.glob("**/lambda_grid_search_progress.json")):
        policy = pf.parent.name
        scenario_path = pf.parent.parent
        scenario_key = scenario_path.relative_to(logs_dir)
        scenarios.setdefault(scenario_key, {})[policy] = pf

    if not scenarios:
        print(f"No lambda_grid_search_progress.json files found under '{logs_dir}'.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(scenarios)} scenario(s), plotting metric: {metric_name}\n")

    for scenario_key in sorted(scenarios):
        policies = scenarios[scenario_key]
        policy_names = sorted(policies)
        n_policies = len(policy_names)
        scenario_label = str(scenario_key)

        # Load all data and build a per-policy norm
        all_data = {}
        policy_norms = {}
        for policy in policy_names:
            grid, val_m, test_m = load_grid_data(policies[policy])
            all_data[policy] = (grid, val_m, test_m)
            policy_norms[policy] = make_norm(
                [val_m[:, args.metric], test_m[:, args.metric]],
                args.max_value, gamma=args.gamma,
            )

        # --- One figure per split (Val / Test) ---
        for split, split_idx in [("Val", 1), ("Test", 2)]:
            fig, axes = plt.subplots(1, n_policies, figsize=(5.5 * n_policies, 5.5))
            if n_policies == 1:
                axes = [axes]

            for col_idx, policy in enumerate(policy_names):
                grid, val_metrics, test_metrics = all_data[policy]
                metric_data = val_metrics[:, args.metric] if split == "Val" else test_metrics[:, args.metric]
                n_pts = len(grid)
                s = point_size_for_grid(n_pts)

                _draw_ternary_on_ax(
                    axes[col_idx], grid, metric_data,
                    title=policy, labels=["λ0", "λ1", "λ2"],
                    s=s, norm=policy_norms[policy],
                )

            plt.tight_layout()
            out_path = save_figure(fig, scenario_label, split, metric_name,
                                   logs_dir, scenario_key, args.output_dir, args.dpi)
            print(f"  [{scenario_label}] {split:4s}  {n_policies} policy/ies  →  {out_path}")
            plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
