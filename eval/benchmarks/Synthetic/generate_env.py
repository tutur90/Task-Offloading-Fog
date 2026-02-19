#!/usr/bin/env python3
"""
Generate a coherent IoT edge-fog-cloud environment (config.json)
and associated task datasets (trainset.csv, testset.csv).

Usage:
    python generate_env.py --num-tasks 30000
    python generate_env.py --num-nodes 8
    python generate_env.py --num-tasks 10000 --fog-ratio 0.7 --cloud-ratio 0.3
    python generate_env.py --num-nodes 12 --tasks-per-node 2000 --seed 42
"""

import argparse
import json
import math
import os
import random
import numpy as np
import pandas as pd

# ─── Fictional country: Lunaria ─────────────────────────────────────────────
COUNTRY_NAME = "Lunaria"
COUNTRY_CENTER = (40.0, 55.0)
COUNTRY_RADIUS = 5.0

CITY_NAMES = [
    "Arenis", "Belvara", "Cindrath", "Dormath", "Eryndel",
    "Falmere", "Gorthyn", "Halvaren", "Ithrand", "Juvalis",
    "Kelmora", "Lundarth", "Morvane", "Netharis", "Orenthal",
    "Pyrathis", "Quelden", "Raventh", "Silvane", "Tormalis",
    "Ulvaren", "Velmora", "Wyndrath", "Xaloren", "Ysmeral",
    "Zephyral", "Aethon", "Brinthas", "Corvale", "Duskara",
    "Elthorne", "Frostmere", "Galvyn", "Haldris", "Isenrath",
    "Jormund", "Kraveth", "Lyndara", "Morthane", "Nytharis",
]

CLOUD_LOCATIONS = [
    {"name": "Singapore",            "lat": 1.2779,   "lon": 103.848},
    {"name": "Saint-Ghislain, Belgium", "lat": 50.4738, "lon": 3.8038},
    {"name": "Iowa, USA",            "lat": 41.878,   "lon": -93.098},
    {"name": "Tokyo, Japan",         "lat": 35.6762,  "lon": 139.6503},
    {"name": "Sydney, Australia",    "lat": -33.8688, "lon": 151.2093},
    {"name": "Frankfurt, Germany",   "lat": 50.1109,  "lon": 8.6821},
    {"name": "São Paulo, Brazil",    "lat": -23.5505, "lon": -46.6333},
    {"name": "Mumbai, India",        "lat": 19.0760,  "lon": 72.8777},
]

# ─── Node hardware specs (ranges) ───────────────────────────────────────────
EDGE_SPEC = {
    "MaxCpuFreq": (8000, 12000),
    "MaxBufferSize": [2048, 3072, 4096],
    "IdleEnergyCoef": (2.0, 5.0),
    "ExeEnergyCoef": (8.0, 15.0),
}
FOG_SPEC = {
    "MaxCpuFreq": (50000, 120000),
    "MaxBufferSize": [6144, 8192, 10240, 12288, 16384],
    "IdleEnergyCoef": (15.0, 35.0),
    "ExeEnergyCoef": (50.0, 160.0),
}
CLOUD_SPEC = {
    "MaxCpuFreq": (400000, 600000),
    "MaxBufferSize": [40960, 51200, 61440],
    "IdleEnergyCoef": (200.0, 350.0),
    "ExeEnergyCoef": (900.0, 1200.0),
}

# ─── Bandwidth ranges (Mbps) ────────────────────────────────────────────────
BW_EDGE_TO_FOG   = (1000, 2500)
BW_FOG_TO_EDGE   = (700, 1700)
BW_EDGE_TO_CLOUD = (2500, 4000)

# ─── Task statistics (reference: 30k tasks / 8 nodes) ───────────────────────
TASK_STATS = {
    "GenerationTime": {"min": 0, "max": 3780},
    "TaskSize":       {"mean": 206, "std": 110, "min": 80,  "max": 300},  # std inflated to compensate snap(10) + truncation
    "TransBitRate":   {"mean": 88,  "std": 50,  "min": 20,  "max": 150},  # std inflated to compensate snap(10) + truncation
    "DDL":            {"mean": 60,  "std": 28,  "min": 20,  "max": 99},
}

DATA_TYPES        = ["Bulk", "LocationBased", "Medical", "Abrupt", "SmallTextual", "Large", "Multimedia"]
DATA_TYPE_WEIGHTS = [0.27,   0.13,            0.13,      0.07,     0.13,           0.13,    0.14]

DEVICE_TYPES        = ["Nodes", "Acuator", "DumbObjects", "Mobile", "Sensor"]
DEVICE_TYPE_WEIGHTS = [0.27,    0.20,      0.20,          0.20,     0.13]

DEFAULT_FOG_RATIO     = 5 / 7   # ~0.714
DEFAULT_CLOUD_RATIO   = 2 / 7   # ~0.286
DEFAULT_TASKS_PER_NODE = 3750


# ─── Helpers ─────────────────────────────────────────────────────────────────

def truncated_normal(mean, std, low, high, size):
    samples = []
    while len(samples) < size:
        batch = np.random.normal(mean, std, size * 3)
        valid = batch[(batch >= low) & (batch <= high)]
        samples.extend(valid.tolist())
    return np.array(samples[:size])


def generate_cycles_per_bit(size):
    """Mixture distribution matching skewed CyclesPerBit (25%=100, 50%=200, 75%=700)."""
    samples = np.empty(size)
    n_low  = int(size * 0.55)
    n_mid  = int(size * 0.25)
    n_high = size - n_low - n_mid

    samples[:n_low]                = truncated_normal(120, 50, 50, 225, n_low)
    samples[n_low:n_low + n_mid]   = truncated_normal(400, 150, 200, 700, n_mid)
    samples[n_low + n_mid:]        = truncated_normal(900, 150, 700, 1200, n_high)
    np.random.shuffle(samples)
    return samples


def snap(arr, step):
    return (np.round(arr / step) * step).astype(int)


# ─── Node generation ────────────────────────────────────────────────────────

def _make_node(device_type, name, node_id, spec, location, lat, lon):
    return {
        "DeviceType":     device_type,
        "NodeType":       "Node",
        "NodeName":       name,
        "NodeId":         node_id,
        "MaxCpuFreq":     random.randint(*spec["MaxCpuFreq"]),
        "MaxBufferSize":  random.choice(spec["MaxBufferSize"]),
        "IdleEnergyCoef": round(random.uniform(*spec["IdleEnergyCoef"]), 1),
        "ExeEnergyCoef":  round(random.uniform(*spec["ExeEnergyCoef"]), 1),
        "LocX":           round(lat, 5),
        "LocY":           round(lon, 5),
        "Location":       location,
    }


def generate_nodes(num_fog, num_cloud):
    nodes = []
    nid = 0
    available_cities = list(CITY_NAMES)
    random.shuffle(available_cities)

    # Edge (always 1)
    city = available_cities.pop()
    lat = COUNTRY_CENTER[0] + random.uniform(-1, 1)
    lon = COUNTRY_CENTER[1] + random.uniform(-1, 1)
    nodes.append(_make_node("Edge", "e0", nid, EDGE_SPEC, f"{city}, {COUNTRY_NAME}", lat, lon))
    nid += 1

    # Fog nodes
    for i in range(num_fog):
        city = available_cities.pop() if available_cities else f"FogCity{i}"
        lat = COUNTRY_CENTER[0] + random.uniform(-COUNTRY_RADIUS, COUNTRY_RADIUS)
        lon = COUNTRY_CENTER[1] + random.uniform(-COUNTRY_RADIUS, COUNTRY_RADIUS)
        nodes.append(_make_node("Fog", f"f{i}", nid, FOG_SPEC, f"{city}, {COUNTRY_NAME}", lat, lon))
        nid += 1

    # Cloud nodes
    cloud_locs = random.sample(CLOUD_LOCATIONS, min(num_cloud, len(CLOUD_LOCATIONS)))
    for i in range(num_cloud):
        loc = cloud_locs[i % len(cloud_locs)]
        lat = loc["lat"] + random.uniform(-0.05, 0.05)
        lon = loc["lon"] + random.uniform(-0.05, 0.05)
        nodes.append(_make_node("Cloud", f"c{i}", nid, CLOUD_SPEC, loc["name"], lat, lon))
        nid += 1

    return nodes


# ─── Edge generation ────────────────────────────────────────────────────────

def generate_edges(nodes):
    edges = []
    edge_id = 0  # e0 always NodeId=0

    for node in nodes[1:]:
        if node["DeviceType"] == "Fog":
            # Bidirectional asymmetric
            edges.append({
                "EdgeType":  "SingleLink",
                "SrcNodeID": edge_id,
                "DstNodeID": node["NodeId"],
                "Bandwidth": random.randint(*BW_EDGE_TO_FOG),
            })
            edges.append({
                "EdgeType":  "SingleLink",
                "SrcNodeID": node["NodeId"],
                "DstNodeID": edge_id,
                "Bandwidth": random.randint(*BW_FOG_TO_EDGE),
            })
        elif node["DeviceType"] == "Cloud":
            edges.append({
                "EdgeType":  "Link",
                "SrcNodeID": edge_id,
                "DstNodeID": node["NodeId"],
                "Bandwidth": random.randint(*BW_EDGE_TO_CLOUD),
            })

    return edges


# ─── Task generation ────────────────────────────────────────────────────────

def generate_tasks(num_tasks):
    # Generation time: sorted uniform over [0, max] scaled by num_tasks ratio
    max_time = TASK_STATS["GenerationTime"]["max"]
    gen_times = np.sort(np.random.uniform(0, max_time, num_tasks))

    # Task size: bimodal to match high variance (std≈75) within [80, 300]
    # 40% low cluster + 60% high cluster
    n_low_ts  = int(num_tasks * 0.40)
    n_high_ts = num_tasks - n_low_ts
    ts_low  = truncated_normal(120, 30, 80, 180, n_low_ts)
    ts_high = truncated_normal(260, 30, 180, 300, n_high_ts)
    task_sizes = np.concatenate([ts_low, ts_high])
    np.random.shuffle(task_sizes)
    task_sizes = np.clip(snap(task_sizes, 10),
                         TASK_STATS["TaskSize"]["min"], TASK_STATS["TaskSize"]["max"])

    # CyclesPerBit: custom skewed mixture, snapped to 25
    cycles = generate_cycles_per_bit(num_tasks)
    cycles = np.clip(snap(cycles, 25), 50, 1200)

    # TransBitRate: uniform matches well (mean≈85, std≈37.5 for [20,150]), snapped to 10
    trans_rates = np.random.uniform(
        TASK_STATS["TransBitRate"]["min"], TASK_STATS["TransBitRate"]["max"], num_tasks
    )
    trans_rates = np.clip(snap(trans_rates, 10),
                          TASK_STATS["TransBitRate"]["min"], TASK_STATS["TransBitRate"]["max"])

    # DDL: uniform matches target well (mean≈59.5, std≈22.8 for [20,99])
    ddls = np.random.uniform(
        TASK_STATS["DDL"]["min"], TASK_STATS["DDL"]["max"], num_tasks
    ).astype(int)

    data_types   = np.random.choice(DATA_TYPES,   size=num_tasks, p=DATA_TYPE_WEIGHTS)
    device_types = np.random.choice(DEVICE_TYPES, size=num_tasks, p=DEVICE_TYPE_WEIGHTS)

    return pd.DataFrame({
        "TaskName":       [f"t{i}" for i in range(num_tasks)],
        "GenerationTime": np.round(gen_times, 2),
        "TaskID":         range(num_tasks),
        "TaskSize":       task_sizes,
        "CyclesPerBit":   cycles.astype(float),
        "TransBitRate":   trans_rates,
        "DDL":            ddls,
        "DataType":       data_types,
        "DeviceType":     device_types,
    })


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate IoT edge-fog-cloud environment and task datasets"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--num-tasks", type=int, help="Number of tasks to generate")
    group.add_argument("--num-nodes", type=int, help="Total number of nodes (1 edge + fog + cloud)")

    parser.add_argument("--fog-ratio",   type=float, default=DEFAULT_FOG_RATIO,
                        help=f"Fog fraction of non-edge nodes (default: {DEFAULT_FOG_RATIO:.4f})")
    parser.add_argument("--cloud-ratio", type=float, default=DEFAULT_CLOUD_RATIO,
                        help=f"Cloud fraction of non-edge nodes (default: {DEFAULT_CLOUD_RATIO:.4f})")
    parser.add_argument("--tasks-per-node", type=float, default=DEFAULT_TASKS_PER_NODE,
                        help=f"Tasks-to-node ratio (default: {DEFAULT_TASKS_PER_NODE})")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                        help="Train split fraction (default: 0.7)")
    parser.add_argument("--output-dir",  type=str, default="./output",
                        help="Output directory (default: ./output)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Normalize ratios
    total_r = args.fog_ratio + args.cloud_ratio
    fog_r   = args.fog_ratio   / total_r
    cloud_r = args.cloud_ratio / total_r

    # Derive counts
    if args.num_tasks is not None:
        num_tasks   = args.num_tasks
        total_nodes = max(3, round(num_tasks / args.tasks_per_node))
    else:
        total_nodes = max(3, args.num_nodes)
        num_tasks   = max(10, round(total_nodes * args.tasks_per_node))

    non_edge  = total_nodes - 1
    num_fog   = max(1, round(non_edge * fog_r))
    num_cloud = max(1, non_edge - num_fog)
    if num_fog + num_cloud != non_edge:
        num_fog = non_edge - num_cloud
    total_nodes = 1 + num_fog + num_cloud

    print(f"┌─ Configuration ────────────────────────────────────")
    print(f"│  Nodes : {total_nodes} total  (1 edge, {num_fog} fog, {num_cloud} cloud)")
    print(f"│  Tasks : {num_tasks}  (train {int(num_tasks * args.train_ratio)}"
          f" / test {num_tasks - int(num_tasks * args.train_ratio)})")
    print(f"│  Ratio : fog={fog_r:.3f}  cloud={cloud_r:.3f}")
    print(f"└────────────────────────────────────────────────────")

    # Generate environment
    nodes = generate_nodes(num_fog, num_cloud)
    edges = generate_edges(nodes)
    config = {"Nodes": nodes, "Edges": edges, "BaseLatencyType": "haversine"}

    # Generate tasks
    tasks_df = generate_tasks(num_tasks)

    # Train/test split (chronological — preserves time ordering)
    split_idx = int(num_tasks * args.train_ratio)
    train_df = tasks_df.iloc[:split_idx].reset_index(drop=True)
    test_df  = tasks_df.iloc[split_idx:].reset_index(drop=True)
    test_df["GenerationTime"] = (test_df["GenerationTime"] - test_df["GenerationTime"].iloc[0]).round(2)
    test_df["TaskName"] = [f"t{i}" for i in range(len(test_df))]
    test_df["TaskID"]   = range(len(test_df))

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    train_df.to_csv(os.path.join(args.output_dir, "trainset.csv"), index=False)
    test_df.to_csv(os.path.join(args.output_dir, "testset.csv"),  index=False)

    print(f"\n✓ Saved to {args.output_dir}/")
    print(f"  config.json   ({len(nodes)} nodes, {len(edges)} links)")
    print(f"  trainset.csv  ({len(train_df)} tasks)")
    print(f"  testset.csv   ({len(test_df)} tasks)")

    # Quick stats validation
    print(f"\n─ Task stats validation ─")
    for col in ["TaskSize", "CyclesPerBit", "TransBitRate", "DDL"]:
        s = tasks_df[col]
        print(f"  {col:15s}  mean={s.mean():.0f}  std={s.std():.0f}  "
              f"min={s.min():.0f}  max={s.max():.0f}")


if __name__ == "__main__":
    main()