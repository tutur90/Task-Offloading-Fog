"""
Benchmark inference time (policy.act() only) for a YAML config.

Usage:
    python benchmark_inference.py configs/Pakistan/Tuple100k/DQL/NATE.yaml
    python benchmark_inference.py configs/Pakistan/Tuple100k/DQL/NATE.yaml --n_runs 3
    python benchmark_inference.py configs/Pakistan/Tuple100k/DQL/NATE.yaml --device cpu --output results.csv
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
    """
    Monkey-patch policy.act() to accumulate total decision time.
    Attaches `_act_times` list to the policy.
    """
    policy._act_times = []
    original_act = policy.act

    def timed_act(env, task, train=False):
        t0 = time.perf_counter()
        result = original_act(env, task, train=False)  # always train=False
        policy._act_times.append(time.perf_counter() - t0)
        return result

    policy.act = timed_act


def wrap_obs_timer(policy):
    """
    Monkey-patch policy._make_observation() to accumulate obs extraction time.
    No-op if policy has no _make_observation.
    """
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
    """
    Monkey-patch policy.model.forward() to accumulate pure forward-pass time.
    Attaches `_model_times` list to the policy. No-op if policy has no model.
    """
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

    For DQL/PPO/Heuristics: returns per-call policy.act() durations (obs extraction + model forward).
      - SimPy env.run() is NOT included (pure decision time).
    For GA: act() runs in subprocesses so per-call timing is not possible.
      Returns total run_generation wall time spread uniformly across tasks as an approximation.
      NOTE: GA total time includes simulation overhead and is not directly comparable to DQL/Heuristic times.
    """
    policy._act_times = []

    algo = config.get("algo", config["policy"])

    if algo in GA_ALGOS:
        from utils.GA import run_generation
        t0 = time.perf_counter()
        result = run_generation(config, policy, test_data, train=False)
        total = time.perf_counter() - t0
        result.close()
        # Spread total time uniformly — approximation only (includes simulation overhead)
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


def main():
    parser = argparse.ArgumentParser(description="Benchmark policy.act() inference time for a YAML config.")
    parser.add_argument("config", type=str, help="Path to the YAML config file.")
    parser.add_argument(
        "--device", default="cpu",
        help="Device: 'cpu', 'cuda', 'auto' (default: cpu).",
    )
    parser.add_argument(
        "--n_runs", type=int, default=1,
        help="Number of inference runs for averaging (default: 1).",
    )
    parser.add_argument(
        "--checkpoint", default=None, metavar="PATH",
        help="Path to a specific checkpoint file. If omitted, uses the latest found in logs/; "
             "if none exists, runs with default (untrained) weights.",
    )
    parser.add_argument(
        "--single_core", action="store_true",
        help="Restrict execution to a single CPU core (torch threads=1, GA n_processes=1).",
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

    if not os.path.isfile(args.config):
        print(f"Config not found: {args.config}")
        sys.exit(1)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config["device"] = args.device

    dataset     = config["env"]["dataset"]
    flag        = config["env"]["flag"]
    policy_name = config["policy"]
    algo        = config.get("algo", policy_name)

    if args.single_core and "training" in config:
        config["training"]["n_processes"] = 1

    # Load test data
    test_csv = f"eval/benchmarks/{dataset}/data/{flag}/testset.csv"
    if not os.path.exists(test_csv):
        print(f"Testset not found: {test_csv}")
        sys.exit(1)

    test_data = pd.read_csv(test_csv)
    n_tasks   = len(test_data)

    # Build policy
    set_seed(config.get("seed", 42))
    env = create_env(config)

    needs_checkpoint = "training" in config
    if needs_checkpoint:
        ext  = ".npz" if algo in GA_ALGOS else ".pt"
        ckpt = args.checkpoint or find_latest_checkpoint(dataset, flag, policy_name, ext)
        policy = policies[policy_name](env, config, dataset=test_data)
        if ckpt is not None:
            print(f"Checkpoint: {ckpt}")
            policy.load(ckpt)
        else:
            print("No checkpoint found — using default (untrained) weights.")
    else:
        policy = policies[policy_name](env, config)

    wrap_act_timer(policy)
    wrap_obs_timer(policy)
    wrap_model_timer(policy)

    # Benchmark n_runs times
    all_act_times: list[float] = []
    all_obs_times: list[float] = []
    all_model_times: list[float] = []
    is_ga = algo in GA_ALGOS
    for run_idx in range(args.n_runs):
        set_seed(config.get("seed", 42))
        if hasattr(policy, "_obs_times"):
            policy._obs_times = []
        if hasattr(policy, "_model_times"):
            policy._model_times = []
        act_times, _ = run_inference(config, policy, test_data)
        all_act_times.extend(act_times)
        if hasattr(policy, "_obs_times"):
            all_obs_times.extend(policy._obs_times)
        if hasattr(policy, "_model_times"):
            all_model_times.extend(policy._model_times)
        total_act = sum(act_times)
        per_task_ms = total_act / len(act_times) * 1000 if act_times else 0
        label = "total run_generation (≈)" if is_ga else "total act()"
        print(f"Run {run_idx + 1}/{args.n_runs}: {total_act:.4f}s {label}  |  {per_task_ms:.4f} ms/task")

    # Aggregate stats across all runs
    arr = np.array(all_act_times) * 1000  # convert to ms
    n_calls = len(arr)

    # Avg inter-arrival time from testset GenerationTime (simulation units → ms)
    inter_arrivals = test_data["GenerationTime"].diff().dropna()
    avg_inter_arrival_ms = float(inter_arrivals.mean()) * 1000
    ratio = arr.mean() / avg_inter_arrival_ms if avg_inter_arrival_ms > 0 else float("nan")

    print(f"\n{'='*60}")
    print(f"Config        : {args.config}")
    print(f"Policy        : {policy_name}  ({algo})")
    print(f"Dataset       : {dataset} / {flag}")
    print(f"CPU mode      : {'single core' if args.single_core else 'full (all cores)'}")
    print(f"Test tasks    : {n_tasks}")
    print(f"Runs          : {args.n_runs}  ({n_calls} calls total)")
    if is_ga:
        print(f"⚠ GA timing   : total run_generation wall time ÷ n_tasks (includes simulation overhead)")
        print(f"               NOT directly comparable to DQL/Heuristic per-task act() times.")
    else:
        print(f"Timing scope  : policy.act() only  (obs extraction + model forward, SimPy excluded)")
    print(f"{'─'*60}")
    print(f"Mean per task : {arr.mean():.4f} ms")
    print(f"Std  per task : {arr.std():.4f} ms")
    print(f"Min  per task : {arr.min():.4f} ms")
    print(f"Max  per task : {arr.max():.4f} ms")
    print(f"P50  per task : {np.percentile(arr, 50):.4f} ms")
    print(f"P95  per task : {np.percentile(arr, 95):.4f} ms")
    print(f"P99  per task : {np.percentile(arr, 99):.4f} ms")
    print(f"Total         : {arr.sum() / 1000:.4f} s")
    if all_model_times or all_obs_times:
        mean_act = arr.mean()
        print(f"{'─'*60}")
        if all_obs_times:
            o_arr = np.array(all_obs_times) * 1000
            print(f"  obs extraction mean: {o_arr.mean():.4f} ms ± {o_arr.std():.4f}  ({o_arr.mean()/mean_act*100:.1f}% of act)")
            print(f"  obs extraction P50 : {np.percentile(o_arr, 50):.4f} ms")
            print(f"  obs extraction P95 : {np.percentile(o_arr, 95):.4f} ms")
        if all_model_times:
            m_arr = np.array(all_model_times) * 1000
            print(f"  model.forward mean : {m_arr.mean():.4f} ms ± {m_arr.std():.4f}  ({m_arr.mean()/mean_act*100:.1f}% of act)")
            print(f"  model.forward P50  : {np.percentile(m_arr, 50):.4f} ms")
            print(f"  model.forward P95  : {np.percentile(m_arr, 95):.4f} ms")
        if all_obs_times and all_model_times:
            t_arr = arr - o_arr - m_arr
            print(f"  tensor+argmax mean : {t_arr.mean():.4f} ms ± {t_arr.std():.4f}  ({t_arr.mean()/mean_act*100:.1f}% of act)")
    print(f"{'─'*60}")
    print(f"Avg inter-arr : {avg_inter_arrival_ms:.4f} ms  (testset GenerationTime)")
    print(f"Mean / inter  : {ratio:.4f}x  ({'real-time feasible' if ratio < 1 else 'EXCEEDS inter-arrival'})")
    print(f"{'='*60}")

    if args.output:
        row = {
            "config":          args.config,
            "policy":          policy_name,
            "algo":            algo,
            "dataset":         dataset,
            "flag":            flag,
            "timing_scope":    "run_generation_wall_approx" if is_ga else "policy_act_only",
            "n_tasks":         n_tasks,
            "n_runs":          args.n_runs,
            "mean_ms":         round(float(arr.mean()),  6),
            "std_ms":          round(float(arr.std()),   6),
            "min_ms":          round(float(arr.min()),   6),
            "max_ms":          round(float(arr.max()),   6),
            "p50_ms":          round(float(np.percentile(arr, 50)), 6),
            "p95_ms":          round(float(np.percentile(arr, 95)), 6),
            "p99_ms":          round(float(np.percentile(arr, 99)), 6),
            "total_act_s":         round(float(arr.sum() / 1000), 6),
            "avg_inter_arrival_ms":   round(avg_inter_arrival_ms, 6),
            "mean_over_inter":        round(ratio, 6),
            "obs_mean_ms":            round(float(np.mean(all_obs_times)   * 1000), 6) if all_obs_times   else None,
            "model_forward_mean_ms":  round(float(np.mean(all_model_times) * 1000), 6) if all_model_times else None,
        }
        df_new = pd.DataFrame([row])
        if os.path.exists(args.output):
            df_new.to_csv(args.output, mode="a", header=False, index=False)
        else:
            df_new.to_csv(args.output, index=False)
        print(f"Results appended to {args.output}")


if __name__ == "__main__":
    main()
