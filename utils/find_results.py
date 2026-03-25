import json, glob

files = glob.glob("logs/*/*/*/multi_seed_0_1_2_3_4_5_6_7_summary.json")

header = f"{'Path':<60} {'M0_mean':>12} {'M0_std':>12} {'M1_mean':>12} {'M1_std':>12} {'M2_mean':>12} {'M2_std':>12} {'M3_mean':>12} {'M3_std':>12}"
print(header)
print("-" * len(header))

for f in sorted(files):
    with open(f) as fp:
        d = json.load(fp)
    m = d['test']['mean']
    s = d['test']['std']
    print(f"{f:<60} {m[0]*100:>12.6f} {s[0]*100:>12.6f} {m[1]:>12.6f} {s[1]:>12.6f} {m[2]:>12.6f} {s[2]:>12.6f} {m[3]:>12.6f} {s[3]:>12.6f}")