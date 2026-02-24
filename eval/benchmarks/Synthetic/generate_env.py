#!/usr/bin/env python3
"""
Generate a coherent IoT edge-fog-cloud environment (config.json)
and associated task datasets (trainset.csv, testset.csv).

Task count follows a density of tasks/node/minute, ensuring consistent
proportionality at any scale (up to 200 nodes).

Usage:
    python generate_env.py --num-tasks 30000
    python generate_env.py --num-nodes 8
    python generate_env.py --num-nodes 50 --density 30
    python generate_env.py --num-tasks 10000 --num-nodes 20
    python generate_env.py --num-nodes 100 --fog-ratio 0.8 --cloud-ratio 0.2
"""

import argparse
import json
import math
import os
import random
import numpy as np
import pandas as pd

# ─── Reference constants ────────────────────────────────────────────────────
REF_TASKS         = 30000
REF_NODES         = 8
REF_MAX_TIME_S    = 3780
REF_MAX_TIME_MIN  = REF_MAX_TIME_S / 60.0                          # 63 min
REF_DENSITY       = REF_TASKS / (REF_NODES * REF_MAX_TIME_MIN)     # ~59.52

DEFAULT_FOG_RATIO   = 5 / 7   # ~0.714
DEFAULT_CLOUD_RATIO = 2 / 7   # ~0.286
DEFAULT_SEED        = 42
MAX_NODES           = 200

# ─── Fictional country: Lunaria ─────────────────────────────────────────────
COUNTRY_NAME    = "Lunaria"
COUNTRY_CENTER  = (40.0, 55.0)
COUNTRY_RADIUS  = 5.0

_PREFIXES = [
    "Ar", "Bel", "Cin", "Dor", "Ery", "Fal", "Gor", "Hal", "Ith", "Juv",
    "Kel", "Lun", "Mor", "Neth", "Oren", "Pyr", "Quel", "Rav", "Sil", "Tor",
    "Ulv", "Vel", "Wyn", "Xal", "Ysm", "Zeph", "Aeth", "Brin", "Cor", "Dusk",
    "Elth", "Fros", "Galv", "Hald", "Isen", "Jor", "Krav", "Lyn", "Morth", "Nyth",
    "Ost", "Prim", "Rhen", "Sten", "Thal", "Urd", "Varn", "Wend", "Xer", "Yel",
]
_SUFFIXES = [
    "enis", "vara", "drath", "math", "ndel", "mere", "thyn", "aren", "rand", "alis",
    "mora", "darth", "vane", "aris", "thal", "this", "den", "enth", "ford", "wyn",
    "gard", "stead", "vale", "crest", "holm", "fell", "gate", "march", "shore", "ridge",
]

def _generate_city_names(n):
    names = set()
    for p in _PREFIXES:
        for s in _SUFFIXES:
            names.add(p + s)
            if len(names) >= n:
                return list(names)[:n]
    while len(names) < n:
        names.add(f"City{len(names)}")
    return list(names)[:n]


CLOUD_LOCATIONS = [
    {"name": "Singapore",               "lat": 1.2779,   "lon": 103.848},
    {"name": "Saint-Ghislain, Belgium",  "lat": 50.4738,  "lon": 3.8038},
    {"name": "Iowa, USA",               "lat": 41.878,   "lon": -93.098},
    {"name": "Tokyo, Japan",            "lat": 35.6762,  "lon": 139.6503},
    {"name": "Sydney, Australia",       "lat": -33.8688, "lon": 151.2093},
    {"name": "Frankfurt, Germany",      "lat": 50.1109,  "lon": 8.6821},
    {"name": "São Paulo, Brazil",       "lat": -23.5505, "lon": -46.6333},
    {"name": "Mumbai, India",           "lat": 19.0760,  "lon": 72.8777},
    {"name": "Oregon, USA",             "lat": 45.5944,  "lon": -121.1787},
    {"name": "Dublin, Ireland",         "lat": 53.3498,  "lon": -6.2603},
    {"name": "Seoul, South Korea",      "lat": 37.5665,  "lon": 126.978},
    {"name": "Johannesburg, South Africa", "lat": -26.2041, "lon": 28.0473},
    {"name": "Montreal, Canada",        "lat": 45.5017,  "lon": -73.5673},
    {"name": "Stockholm, Sweden",       "lat": 59.3293,  "lon": 18.0686},
    {"name": "Santiago, Chile",         "lat": -33.4489, "lon": -70.6693},
    {"name": "Taipei, Taiwan",          "lat": 25.033,   "lon": 121.5654},
    {"name": "Warsaw, Poland",          "lat": 52.2297,  "lon": 21.0122},
    {"name": "Doha, Qatar",             "lat": 25.2854,  "lon": 51.531},
    {"name": "Jakarta, Indonesia",      "lat": -6.2088,  "lon": 106.8456},
    {"name": "Helsinki, Finland",       "lat": 60.1699,  "lon": 24.9384},
]

# ─── Node hardware specs ────────────────────────────────────────────────────
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

# ─── Task statistics ────────────────────────────────────────────────────────
TASK_STATS = {
    "TaskSize":     {"min": 80,  "max": 300},
    "TransBitRate": {"min": 20,  "max": 150},
    "DDL":          {"min": 20,  "max": 99},
}

DATA_TYPES        = ["Bulk", "LocationBased", "Medical", "Abrupt", "SmallTextual", "Large", "Multimedia"]
DATA_TYPE_WEIGHTS = [0.27,   0.13,            0.13,      0.07,     0.13,           0.13,    0.14]

DEVICE_TYPES        = ["Nodes", "Acuator", "DumbObjects", "Mobile", "Sensor"]
DEVICE_TYPE_WEIGHTS = [0.27,    0.20,      0.20,          0.20,     0.13]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def truncated_normal(mean, std, low, high, size):
    samples = []
    while len(samples) < size:
        batch = np.random.normal(mean, std, size * 3)
        valid = batch[(batch >= low) & (batch <= high)]
        samples.extend(valid.tolist())
    return np.array(samples[:size])


def generate_cycles_per_bit(size):
    """Mixture distribution: mean≈350, std≈320."""
    samples = np.empty(size)
    n_low  = int(size * 0.55)
    n_mid  = int(size * 0.25)
    n_high = size - n_low - n_mid
    samples[:n_low]              = truncated_normal(120, 50, 50, 225, n_low)
    samples[n_low:n_low + n_mid] = truncated_normal(400, 150, 200, 700, n_mid)
    samples[n_low + n_mid:]      = truncated_normal(900, 150, 700, 1200, n_high)
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
    city_names = _generate_city_names(1 + num_fog)
    random.shuffle(city_names)

    # Edge (always 1)
    city = city_names.pop()
    lat = COUNTRY_CENTER[0] + random.uniform(-1, 1)
    lon = COUNTRY_CENTER[1] + random.uniform(-1, 1)
    nodes.append(_make_node("Edge", "e0", nid, EDGE_SPEC, f"{city}, {COUNTRY_NAME}", lat, lon))
    nid += 1

    # Fog
    for i in range(num_fog):
        city = city_names.pop() if city_names else f"FogCity{i}"
        lat = COUNTRY_CENTER[0] + random.uniform(-COUNTRY_RADIUS, COUNTRY_RADIUS)
        lon = COUNTRY_CENTER[1] + random.uniform(-COUNTRY_RADIUS, COUNTRY_RADIUS)
        nodes.append(_make_node("Fog", f"f{i}", nid, FOG_SPEC, f"{city}, {COUNTRY_NAME}", lat, lon))
        nid += 1

    # Cloud (cycle through real datacenter locations)
    for i in range(num_cloud):
        loc = CLOUD_LOCATIONS[i % len(CLOUD_LOCATIONS)]
        lat = loc["lat"] + random.uniform(-0.05, 0.05)
        lon = loc["lon"] + random.uniform(-0.05, 0.05)
        nodes.append(_make_node("Cloud", f"c{i}", nid, CLOUD_SPEC, loc["name"], lat, lon))
        nid += 1

    return nodes


# ─── Edge generation ────────────────────────────────────────────────────────

def generate_edges(nodes):
    edges = []
    edge_id = 0

    for node in nodes[1:]:
        if node["DeviceType"] == "Fog":
            edges.append({
                "EdgeType": "SingleLink", "SrcNodeID": edge_id,
                "DstNodeID": node["NodeId"], "Bandwidth": random.randint(*BW_EDGE_TO_FOG),
            })
            edges.append({
                "EdgeType": "SingleLink", "SrcNodeID": node["NodeId"],
                "DstNodeID": edge_id, "Bandwidth": random.randint(*BW_FOG_TO_EDGE),
            })
        elif node["DeviceType"] == "Cloud":
            edges.append({
                "EdgeType": "Link", "SrcNodeID": edge_id,
                "DstNodeID": node["NodeId"], "Bandwidth": random.randint(*BW_EDGE_TO_CLOUD),
            })

    return edges


# ─── Task generation ────────────────────────────────────────────────────────

def generate_tasks(num_tasks, max_time_s):
    gen_times = np.sort(np.random.uniform(0, max_time_s, num_tasks))

    # TaskSize: bimodal → mean≈202, std≈69
    n_low_ts  = int(num_tasks * 0.40)
    n_high_ts = num_tasks - n_low_ts
    task_sizes = np.concatenate([
        truncated_normal(120, 30, 80, 180, n_low_ts),
        truncated_normal(260, 30, 180, 300, n_high_ts),
    ])
    np.random.shuffle(task_sizes)
    task_sizes = np.clip(snap(task_sizes, 10), TASK_STATS["TaskSize"]["min"], TASK_STATS["TaskSize"]["max"])

    # CyclesPerBit: skewed mixture → mean≈358, std≈317
    cycles = np.clip(snap(generate_cycles_per_bit(num_tasks), 25), 50, 1200)

    # TransBitRate: uniform → mean≈85, std≈38
    trans_rates = np.clip(
        snap(np.random.uniform(TASK_STATS["TransBitRate"]["min"], TASK_STATS["TransBitRate"]["max"], num_tasks), 10),
        TASK_STATS["TransBitRate"]["min"], TASK_STATS["TransBitRate"]["max"],
    )

    # DDL: uniform → mean≈59, std≈23
    ddls = np.random.uniform(TASK_STATS["DDL"]["min"], TASK_STATS["DDL"]["max"], num_tasks).astype(int)

    return pd.DataFrame({
        "TaskName":       [f"t{i}" for i in range(num_tasks)],
        "GenerationTime": np.round(gen_times, 2),
        "TaskID":         range(num_tasks),
        "TaskSize":       task_sizes,
        "CyclesPerBit":   cycles.astype(float),
        "TransBitRate":   trans_rates,
        "DDL":            ddls,
        "DataType":       np.random.choice(DATA_TYPES, size=num_tasks, p=DATA_TYPE_WEIGHTS),
        "DeviceType":     np.random.choice(DEVICE_TYPES, size=num_tasks, p=DEVICE_TYPE_WEIGHTS),
    })


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate IoT edge-fog-cloud environment and task datasets"
    )
    parser.add_argument("--num-tasks", type=int, default=None,
                        help="Number of tasks (derived from density × nodes if omitted)")
    parser.add_argument("--num-nodes", type=int, default=None,
                        help="Total nodes incl. 1 edge (max 200, derived if omitted)")
    parser.add_argument("--density", type=float, default=None,
                        help=f"Tasks per node per minute (default: {REF_DENSITY:.2f})")
    parser.add_argument("--max-time", type=float, default=REF_MAX_TIME_S,
                        help=f"Max generation time in seconds (default: {REF_MAX_TIME_S})")
    parser.add_argument("--fog-ratio",   type=float, default=DEFAULT_FOG_RATIO,
                        help=f"Fog fraction of non-edge nodes (default: {DEFAULT_FOG_RATIO:.4f})")
    parser.add_argument("--cloud-ratio", type=float, default=DEFAULT_CLOUD_RATIO,
                        help=f"Cloud fraction of non-edge nodes (default: {DEFAULT_CLOUD_RATIO:.4f})")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                        help="Train split fraction (default: 0.7)")
    parser.add_argument("--output-dir",  type=str, default=None,
                        help="Output directory (default: ./data/{N}N{T}T{D}D)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed (default: {DEFAULT_SEED})")

    args = parser.parse_args()
    if args.num_tasks is None and args.num_nodes is None:
        parser.error("At least one of --num-tasks or --num-nodes is required")

    random.seed(args.seed)
    np.random.seed(args.seed)

    density_given = args.density is not None
    density = args.density if density_given else REF_DENSITY

    total_r = args.fog_ratio + args.cloud_ratio
    fog_r   = args.fog_ratio   / total_r
    cloud_r = args.cloud_ratio / total_r
    max_time_min = args.max_time / 60.0

    # ── Derive missing counts from density ──
    if args.num_tasks is not None and args.num_nodes is not None:
        num_tasks   = args.num_tasks
        total_nodes = min(MAX_NODES, max(3, args.num_nodes))
        if density_given:
            # All three specified: derive max_time from num_tasks, num_nodes, density
            max_time_min  = num_tasks / (total_nodes * density)
            args.max_time = max_time_min * 60
        eff_density = num_tasks / (total_nodes * max_time_min)
    elif args.num_tasks is not None:
        num_tasks   = args.num_tasks
        total_nodes = min(MAX_NODES, max(3, round(num_tasks / (density * max_time_min))))
        eff_density = density
    else:
        total_nodes = min(MAX_NODES, max(3, args.num_nodes))
        num_tasks   = max(10, round(density * total_nodes * max_time_min))
        eff_density = density

    # ── Fog / cloud split ──
    non_edge  = total_nodes - 1
    num_fog   = max(1, round(non_edge * fog_r))
    num_cloud = max(1, non_edge - num_fog)
    if num_fog + num_cloud != non_edge:
        num_fog = non_edge - num_cloud
    total_nodes = 1 + num_fog + num_cloud

    if args.output_dir is None:
        tasks_k = num_tasks // 1000
        args.output_dir = f"./data/{total_nodes}N{tasks_k}T{round(eff_density)}D"

    print(f"┌─ Configuration ────────────────────────────────────")
    print(f"│  Nodes   : {total_nodes} total  (1 edge, {num_fog} fog, {num_cloud} cloud)")
    print(f"│  Tasks   : {num_tasks}  (train {int(num_tasks * args.train_ratio)}"
          f" / test {num_tasks - int(num_tasks * args.train_ratio)})")
    print(f"│  Density : {eff_density:.2f} tasks/node/min")
    print(f"│  Duration: {args.max_time:.0f}s ({max_time_min:.1f} min)")
    print(f"│  Ratio   : fog={fog_r:.3f}  cloud={cloud_r:.3f}")
    print(f"│  Seed    : {args.seed}")
    print(f"└────────────────────────────────────────────────────")

    nodes  = generate_nodes(num_fog, num_cloud)
    edges  = generate_edges(nodes)
    config = {"Nodes": nodes, "Edges": edges, "BaseLatencyType": "haversine"}

    tasks_df = generate_tasks(num_tasks, args.max_time)

    split_idx = int(num_tasks * args.train_ratio)
    train_df  = tasks_df.iloc[:split_idx].reset_index(drop=True)
    test_df   = tasks_df.iloc[split_idx:].reset_index(drop=True)
    test_df["GenerationTime"] = (test_df["GenerationTime"] - test_df["GenerationTime"].iloc[0]).round(2)
    test_df["TaskName"] = [f"t{i}" for i in range(len(test_df))]
    test_df["TaskID"]   = range(len(test_df))

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    train_df.to_csv(os.path.join(args.output_dir, "trainset.csv"), index=False)
    test_df.to_csv(os.path.join(args.output_dir, "testset.csv"),  index=False)

    print(f"\n✓ Saved to {args.output_dir}/")
    print(f"  config.json   ({len(nodes)} nodes, {len(edges)} links)")
    print(f"  trainset.csv  ({len(train_df)} tasks)")
    print(f"  testset.csv   ({len(test_df)} tasks)")

    print(f"\n─ Task stats validation ─")
    for col in ["TaskSize", "CyclesPerBit", "TransBitRate", "DDL"]:
        s = tasks_df[col]
        print(f"  {col:15s}  mean={s.mean():.0f}  std={s.std():.0f}  "
              f"min={s.min():.0f}  max={s.max():.0f}")


if __name__ == "__main__":
    main()