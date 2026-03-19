"""
Benchmark inference time (policy.act() only) for one or more YAML configs.

Usage:
    python benchmark_inference.py configs/Pakistan/Tuple100k/DQL/T-NATE.yaml
    python benchmark_inference.py configs/**/T-*.yaml --single_core
    python benchmark_inference.py configs/Pakistan/Tuple100k/DQL/T-NATE.yaml --n_runs 3 --warmup 100 --output results.csv
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

from policies import policies
from utils.dql import run_epoch
from utils.utils import create_env, set_seed

GA_ALGOS  = ["NPGA", "NSGA2"]
PPO_ALGOS = ["PPO"]


def find_latest_checkpoint(dataset: str, flag: str, policy_name: str, ext: str = ".pt") -> str | None:
    """Return the most recent checkpoint file path, or None."""
    base = os.path.join("logs", dataset, flag, policy_name)
    if not os.path.isdir(base):
        return None
    pattern = os.path.join(base, "*", "checkpoints", f"checkpoint_epoch_*{ext}")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def wrap_act_timer(policy):
    policy._act_times = []
    original_act = policy.act

    def timed_act(env, task, train=False):
        t0 = time.perf_counter()
        result = original_act(env, task, train=False)
        policy._act_times.append(time.perf_counter() - t0)
        return result

    policy.act = timed_act


def wrap_obs_timer(policy):
    if not hasattr(policy, "_make_observation"):
        return
    policy._obs_times = []
    original_obs = policy._make_observation

    def timed_obs(*args, **kwargs):
        t0 = time.perf_counter()
        result = original_obs(*args, **kwargs)
        policy._obs_times.append(time.perf_counter() - t0)
        return result

    policy._make_observation = timed_obs


def wrap_model_timer(policy):
    if not hasattr(policy, "model"):
        return
    policy._model_times = []
    original_forward = policy.model.forward

    def timed_forward(*args, **kwargs):
        t0 = time.perf_counter()
        result = original_forward(*args, **kwargs)
        policy._model_times.append(time.perf_counter() - t0)
        return result

    policy.model.forward = timed_forward


def run_inference(config: dict, policy, test_data: pd.DataFrame) -> tuple[list[float], bool]:
    """
    Run one inference epoch and return (act_times, is_ga).
    SimPy env.run() time is NOT included (pure decision time).
    """
    policy._act_times = []

    algo = config.get("algo", config["policy"])

    if algo in GA_ALGOS:
        from utils.GA import run_generation
        t0 = time.perf_counter()
        result = run_generation(config, policy, test_data, train=False)
        total = time.perf_counter() - t0
        result.close()
        n = len(test_data)
        act_times = [total / n] * n
        return act_times, True

    elif algo in PPO_ALGOS:
        from utils.ppo import run_epoch_ppo
        env = run_epoch_ppo(config, policy, test_data, train=False)
        env.close()
    else:
        env = run_epoch(config, policy, test_data, train=False)
        env.close()

    return list(policy._act_times), False


def iqr(a: np.ndarray) -> float:
    return float(np.percentile(a, 75) - np.percentile(a, 25))


def stats(a: np.ndarray) -> dict:
    return {
        "median": float(np.median(a)),
        "iqr":    iqr(a),
        "min":    float(a.min()),
        "max":    float(a.max()),
        "p95":    float(np.percentile(a, 95)),
        "p99":    float(np.percentile(a, 99)),
        "total":  float(a.sum()),
    }


def build_policy(args, config, env, test_data):
    algo        = config.get("algo", config["policy"])
    policy_name = config["policy"]
    dataset     = config["env"]["dataset"]
    flag        = config["env"]["flag"]

    needs_checkpoint = "training" in config and not args.no_checkpoint
    if needs_checkpoint:
        ext  = ".npz" if algo in GA_ALGOS else ".pt"
        ckpt = args.checkpoint or find_latest_checkpoint(dataset, flag, policy_name, ext)
        policy = policies[policy_name](env, config, dataset=test_data)
        if ckpt is not None:
            print(f"  Checkpoint: {ckpt}")
            policy.load(ckpt)
        else:
            print("  No checkpoint found — using default (untrained) weights.")
    elif "training" in config and args.no_checkpoint:
        policy = policies[policy_name](env, config, dataset=test_data)
        print("  --no_checkpoint: using default (untrained) weights.")
    else:
        policy = policies[policy_name](env, config)

    return policy


def benchmark_one(config_path: str, args) -> dict | None:
    """Run the full benchmark for one config. Returns a result dict."""
    if not os.path.isfile(config_path):
        print(f"Config not found: {config_path}")
        return None

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["device"] = args.device

    dataset     = config["env"]["dataset"]
    flag        = config["env"]["flag"]
    policy_name = config["policy"]
    algo        = config.get("algo", policy_name)

    if args.single_core and "training" in config:
        config["training"]["n_processes"] = 1

    test_csv = f"eval/benchmarks/{dataset}/data/{flag}/testset.csv"
    if not os.path.exists(test_csv):
        print(f"Testset not found: {test_csv}")
        return None

    test_data = pd.read_csv(test_csv)
    n_tasks   = len(test_data)

    set_seed(config.get("seed", 42))
    env    = create_env(config)
    policy = build_policy(args, config, env, test_data)

    wrap_act_timer(policy)
    wrap_obs_timer(policy)
    wrap_model_timer(policy)

    all_act_times:   list[float] = []
    all_obs_times:   list[float] = []
    all_model_times: list[float] = []
    is_ga = algo in GA_ALGOS

    for run_idx in range(args.n_runs):
        set_seed(config.get("seed", 42))
        if hasattr(policy, "_obs_times"):
            policy._obs_times = []
        if hasattr(policy, "_model_times"):
            policy._model_times = []

        act_times, _ = run_inference(config, policy, test_data)

        # trim warmup
        act_times = act_times[args.warmup:]
        obs_t  = (policy._obs_times[args.warmup:]   if hasattr(policy, "_obs_times")   else [])
        mod_t  = (policy._model_times[args.warmup:]  if hasattr(policy, "_model_times") else [])

        all_act_times.extend(act_times)
        all_obs_times.extend(obs_t)
        all_model_times.extend(mod_t)

        total_act    = sum(act_times)
        per_task_ms  = total_act / len(act_times) * 1000 if act_times else 0
        label = "total run_generation (≈)" if is_ga else "total act()"
        print(f"  Run {run_idx + 1}/{args.n_runs}: {total_act:.4f}s {label}  |  {per_task_ms:.4f} ms/task")

    arr    = np.array(all_act_times) * 1000
    n_calls = len(arr)
    s      = stats(arr)

    inter_arrivals      = test_data["GenerationTime"].diff().dropna()
    avg_inter_arrival_ms = float(inter_arrivals.mean()) * 1000
    ratio = s["median"] / avg_inter_arrival_ms if avg_inter_arrival_ms > 0 else float("nan")

    result = {
        "config":               config_path,
        "policy":               policy_name,
        "algo":                 algo,
        "dataset":              dataset,
        "flag":                 flag,
        "timing_scope":         "run_generation_wall_approx" if is_ga else "policy_act_only",
        "n_tasks":              n_tasks,
        "warmup":               args.warmup,
        "n_runs":               args.n_runs,
        "n_calls":              n_calls,
        "median_ms":            round(s["median"], 6),
        "iqr_ms":               round(s["iqr"],    6),
        "min_ms":               round(s["min"],     6),
        "max_ms":               round(s["max"],     6),
        "p95_ms":               round(s["p95"],     6),
        "p99_ms":               round(s["p99"],     6),
        "total_act_s":          round(s["total"] / 1000, 6),
        "avg_inter_arrival_ms": round(avg_inter_arrival_ms, 6),
        "median_over_inter":    round(ratio, 6),
        "obs_median_ms":        round(float(np.median(np.array(all_obs_times)   * 1000)), 6) if all_obs_times   else None,
        "obs_iqr_ms":           round(float(iqr(np.array(all_obs_times)         * 1000)), 6) if all_obs_times   else None,
        "model_median_ms":      round(float(np.median(np.array(all_model_times) * 1000)), 6) if all_model_times else None,
        "model_iqr_ms":         round(float(iqr(np.array(all_model_times)       * 1000)), 6) if all_model_times else None,
        # raw arrays for breakdown
        "_arr":         arr,
        "_obs_arr":     np.array(all_obs_times)   * 1000 if all_obs_times   else None,
        "_model_arr":   np.array(all_model_times) * 1000 if all_model_times else None,
        "_is_ga":       is_ga,
    }
    return result


def print_single(r: dict, config_path: str, args):
    arr       = r["_arr"]
    o_arr     = r["_obs_arr"]
    m_arr     = r["_model_arr"]
    is_ga     = r["_is_ga"]

    print(f"\n{'='*62}")
    print(f"Config        : {config_path}")
    print(f"Policy        : {r['policy']}  ({r['algo']})")
    print(f"Dataset       : {r['dataset']} / {r['flag']}")
    print(f"CPU mode      : {'single core' if args.single_core else 'full (all cores)'}")
    print(f"Test tasks    : {r['n_tasks']}  (warmup trimmed: {r['warmup']})")
    print(f"Runs          : {r['n_runs']}  ({r['n_calls']} calls total)")
    if is_ga:
        print(f"⚠ GA timing   : total run_generation wall time ÷ n_tasks (includes simulation overhead)")
        print(f"               NOT directly comparable to DQL/Heuristic per-task act() times.")
    else:
        print(f"Timing scope  : policy.act() only  (obs extraction + model forward, SimPy excluded)")
    print(f"{'─'*62}")
    print(f"Median per task : {r['median_ms']:.4f} ms  ±IQR {r['iqr_ms']:.4f} ms")
    print(f"Min    per task : {r['min_ms']:.4f} ms")
    print(f"Max    per task : {r['max_ms']:.4f} ms")
    print(f"P95    per task : {r['p95_ms']:.4f} ms")
    print(f"P99    per task : {r['p99_ms']:.4f} ms")
    print(f"Total           : {r['total_act_s']:.4f} s")

    if o_arr is not None or m_arr is not None:
        print(f"{'─'*62}")
        if o_arr is not None:
            pct = np.median(o_arr) / r['median_ms'] * 100
            print(f"  obs extraction  median: {np.median(o_arr):.4f} ms  ±IQR {iqr(o_arr):.4f}  ({pct:.1f}% of act)")
            print(f"  obs extraction  P95   : {np.percentile(o_arr, 95):.4f} ms")
        if m_arr is not None:
            pct = np.median(m_arr) / r['median_ms'] * 100
            print(f"  model.forward   median: {np.median(m_arr):.4f} ms  ±IQR {iqr(m_arr):.4f}  ({pct:.1f}% of act)")
            print(f"  model.forward   P95   : {np.percentile(m_arr, 95):.4f} ms")
        if o_arr is not None and m_arr is not None:
            t_arr = arr - o_arr - m_arr
            pct = np.median(t_arr) / r['median_ms'] * 100
            print(f"  tensor+argmax   median: {np.median(t_arr):.4f} ms  ±IQR {iqr(t_arr):.4f}  ({pct:.1f}% of act)")

    print(f"{'─'*62}")
    print(f"Avg inter-arr   : {r['avg_inter_arrival_ms']:.4f} ms  (testset GenerationTime)")
    ratio = r["median_over_inter"]
    print(f"Median / inter  : {ratio:.4f}x  ({'real-time feasible' if ratio < 1 else 'EXCEEDS inter-arrival'})")
    print(f"{'='*62}")


def print_table(results: list[dict]):
    """Print a compact comparison table for all benchmarked configs, grouped by directory."""
    dir_w  = max(len(os.path.dirname(r["config"])) for r in results)
    dir_w  = max(dir_w, 7)
    pol_w  = max(len(r["policy"]) for r in results)
    pol_w  = max(pol_w, 6)

    header = (
        f"{'Dir':<{dir_w}}  {'Policy':<{pol_w}} {'Act med':>8} {'Act IQR':>8} "
        f"{'Fwd med':>8} {'Fwd IQR':>8} {'Tensor med':>11} {'Med/inter':>10}"
    )
    sep = "─" * len(header)

    # Sort by directory then policy name
    results_sorted = sorted(results, key=lambda r: (os.path.dirname(r["config"]), r["policy"]))

    print(f"\n{'='*len(header)}")
    print("COMPARISON TABLE  (all times in ms)")
    print(sep)
    print(header)
    print(sep)

    prev_dir = None
    for r in results_sorted:
        cur_dir = os.path.dirname(r["config"])
        if prev_dir is not None and cur_dir != prev_dir:
            print(sep)
        prev_dir = cur_dir

        fwd_med  = f"{r['model_median_ms']:.3f}" if r["model_median_ms"] is not None else "  —"
        fwd_iqr  = f"{r['model_iqr_ms']:.3f}"    if r["model_iqr_ms"]   is not None else "  —"

        # tensor+argmax = act - obs - model  (median approximation)
        if r["obs_median_ms"] is not None and r["model_median_ms"] is not None:
            tensor_med = f"{r['median_ms'] - r['obs_median_ms'] - r['model_median_ms']:.3f}"
        else:
            tensor_med = "  —"

        feasible = "✓" if r["median_over_inter"] < 1 else "✗"
        print(
            f"{cur_dir:<{dir_w}}  "
            f"{r['policy']:<{pol_w}} "
            f"{r['median_ms']:>8.3f} "
            f"{r['iqr_ms']:>8.3f} "
            f"{fwd_med:>8} "
            f"{fwd_iqr:>8} "
            f"{tensor_med:>11} "
            f"{r['median_over_inter']:>8.4f}x {feasible}"
        )
    print(f"{'='*len(header)}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark policy.act() inference time.")
    parser.add_argument(
        "config", nargs="+",
        help="One or more YAML config file paths (shell glob expansion supported, e.g. configs/**/T-*.yaml).",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device: 'cpu', 'cuda', 'auto' (default: cpu).",
    )
    parser.add_argument(
        "--n_runs", type=int, default=1,
        help="Number of inference runs per config (default: 1).",
    )
    parser.add_argument(
        "--warmup", type=int, default=500, metavar="N",
        help="Number of first-N iterations to discard as warmup (default: 500).",
    )
    parser.add_argument(
        "--checkpoint", default=None, metavar="PATH",
        help="Path to a specific checkpoint file.",
    )
    parser.add_argument(
        "--no_checkpoint", action="store_true",
        help="Skip checkpoint loading and use default (untrained) weights.",
    )
    parser.add_argument(
        "--single_core", action="store_true",
        help="Restrict to a single CPU core (torch threads=1, GA n_processes=1).",
    )
    parser.add_argument(
        "--output", default=None, metavar="CSV",
        help="Append results to a CSV file.",
    )
    args = parser.parse_args()

    if args.single_core:
        import torch
        torch.set_num_threads(1)
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"

    # Expand any glob patterns that the shell didn't expand (e.g. quoted globs)
    config_paths = []
    for pat in args.config:
        if any(c in pat for c in ("*", "?", "[")):
            expanded = sorted(glob.glob(pat, recursive=True))
            if not expanded:
                print(f"Warning: no files matched pattern '{pat}'")
            config_paths.extend(expanded)
        else:
            config_paths.append(pat)

    results = []
    for cp in config_paths:
        print(f"── {cp}")
        r = benchmark_one(cp, args)
        if r is not None:
            results.append(r)
            print_single(r, cp, args)

    if len(results) > 1:
        print_table(results)

    if args.output and results:
        _save_csv(results, args.output)


def _save_csv(results: list[dict], path: str):
    """Append results (excluding raw arrays) to a CSV."""
    skip = {"_arr", "_obs_arr", "_model_arr", "_is_ga"}
    rows = [{k: v for k, v in r.items() if k not in skip} for r in results]
    df_new = pd.DataFrame(rows)
    if os.path.exists(path):
        df_new.to_csv(path, mode="a", header=False, index=False)
    else:
        df_new.to_csv(path, index=False)
    print(f"Results saved to {path}")


if __name__ == "__main__":
    main()
