import argparse, json, os, re, glob, math
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional

MP4_PATTERN = re.compile(r"episode=(\d+).*?success=(True|False)", re.IGNORECASE)
EP_FROM_JSON_PATTERN = re.compile(r"ep_(\d+)_(\d+)", re.IGNORECASE)  # ep_{suite}_{episode}
KEY_SERIES = "swapratio_pair_t"
KEY_T = "T"

def build_episode_success_map(rollout_dir: str) -> Dict[int, bool]:
    ep2success = {}
    for path in glob.glob(os.path.join(rollout_dir, "*.mp4")):
        fn = os.path.basename(path)
        m = MP4_PATTERN.search(fn)
        if not m:
            continue
        ep = int(m.group(1))
        succ = m.group(2).lower() == "true"
        ep2success[ep] = succ
    if not ep2success:
        raise RuntimeError(f"No success signal of any episode under {rollout_dir}.")
    return ep2success

def find_all_jsons(results_dir: str) -> List[str]:
    pats = [
        os.path.join(results_dir, "ep_*_*_two_metrics.json"),
        os.path.join(results_dir, "ep_*_*.json"),
    ]
    files = []
    for p in pats:
        files.extend(glob.glob(p))
    return sorted(set(files))

def parse_suite_episode_from_json(json_path: str) -> Tuple[int, int, Dict[str, Any]]:
    data = None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = None

    if isinstance(data, dict) and isinstance(data.get("episode"), str):
        m = EP_FROM_JSON_PATTERN.fullmatch(data["episode"])
        if m:
            return int(m.group(1)), int(m.group(2)), data

    fn = os.path.basename(json_path)
    m2 = re.search(r"ep_(\d+)_(\d+)", fn, re.IGNORECASE)
    if not m2:
        raise ValueError(f"Cannot parse suite/episode from {json_path}.")
    if data is None:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    return int(m2.group(1)), int(m2.group(2)), data

def to_float_list(x) -> List[float]:
    if not isinstance(x, list):
        return []
    out = []
    for v in x:
        try:
            out.append(float(v))
        except Exception:
            out.append(float("nan"))
    return out

def get_int_T(data: Dict[str, Any]) -> Optional[int]:
    if KEY_T not in data:
        return None
    try:
        return int(data[KEY_T])
    except Exception:
        try:
            return int(float(data[KEY_T]))
        except Exception:
            return None

def main():
    ap = argparse.ArgumentParser(description="Plot success vs failure mean swapratio over time using global min T; optional std bands.")
    ap.add_argument("--results_base", type=str, default="../results")
    ap.add_argument("--rollouts_base", type=str, default="../rollouts")
    ap.add_argument("--suite_name", type=str, default="libero_spatial")
    ap.add_argument("--sd", type=str, default="7")
    ap.add_argument("--episodes_per_suite", type=int, default=50, help="Episodes per suite to compute global episode id")
    ap.add_argument("--title", type=str, default="Libero Spatial")
    ap.add_argument("--save", type=str, default="../plots/swp", help="If set, save figure to this path")
    ap.add_argument("--std", action="store_true", help="Also compute std and draw lighter shaded bands")
    args = ap.parse_args()

    ep2succ = build_episode_success_map(os.path.join(args.rollouts_base, args.suite_name, 'seed_'+args.sd))
    json_files = find_all_jsons(os.path.join(args.results_base, args.suite_name, 'seed_'+args.sd))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found under {os.path.join(args.results_base, args.suite_name, 'seed_'+args.sd)}.")

    # Gather episodes and find global min T
    episodes = []  # (succ_flag: bool, T: int, series: List[float])
    min_T = None
    for jp in json_files:
        suite_i, ep_in_suite, data = parse_suite_episode_from_json(jp)
        global_ep = (suite_i - 1) * args.episodes_per_suite + ep_in_suite
        succ_flag = ep2succ.get(global_ep)
        if succ_flag is None:
            continue

        T = get_int_T(data)
        if T is None or T <= 0:
            continue

        series = to_float_list(data.get(KEY_SERIES, []))
        if not series:
            continue

        episodes.append((succ_flag, T, series))
        if min_T is None or T < min_T:
            min_T = T

    if min_T is None:
        raise RuntimeError("No usable episodes with valid T and swapratio series were found.")

    # Accumulators
    succ_sum = [0.0] * min_T
    succ_cnt = [0] * min_T
    fail_sum = [0.0] * min_T
    fail_cnt = [0] * min_T

    succ_sumsq = [0.0] * min_T if args.std else None
    fail_sumsq = [0.0] * min_T if args.std else None

    succ_eps = 0
    fail_eps = 0

    for succ_flag, T, series in episodes:
        use_len = min(min_T, len(series))
        if succ_flag:
            succ_eps += 1
            for i in range(use_len):
                v = series[i]
                if v == v:
                    succ_sum[i] += v
                    succ_cnt[i] += 1
                    if args.std:
                        succ_sumsq[i] += v * v
        else:
            fail_eps += 1
            for i in range(use_len):
                v = series[i]
                if v == v:
                    fail_sum[i] += v
                    fail_cnt[i] += 1
                    if args.std:
                        fail_sumsq[i] += v * v

    # Means
    succ_mean = [ (succ_sum[i] / succ_cnt[i]) if succ_cnt[i] > 0 else float("nan") for i in range(min_T) ]
    fail_mean = [ (fail_sum[i] / fail_cnt[i]) if fail_cnt[i] > 0 else float("nan") for i in range(min_T) ]

    # Optional std (sample std, ddof=1)
    if args.std:
        succ_std = []
        for i in range(min_T):
            n = succ_cnt[i]
            if n >= 2:
                var = (succ_sumsq[i] - n * (succ_mean[i] ** 2)) / (n - 1)
                succ_std.append(math.sqrt(var) if var > 0 else 0.0)
            else:
                succ_std.append(float("nan"))
        fail_std = []
        for i in range(min_T):
            n = fail_cnt[i]
            if n >= 2:
                var = (fail_sumsq[i] - n * (fail_mean[i] ** 2)) / (n - 1)
                fail_std.append(math.sqrt(var) if var > 0 else 0.0)
            else:
                fail_std.append(float("nan"))
    else:
        succ_std = fail_std = None

    # Plot
    t = list(range(1, min_T + 1))
    plt.figure(figsize=(9, 5.2))

    # Draw mean lines (darker) first to get their colors
    line_succ, = plt.plot(t, succ_mean, "-",  linewidth=2.2, label=f"Success")
    line_fail, = plt.plot(t, fail_mean, "--", linewidth=2.2, label=f"Failure")

    # If std: draw lighter shaded area using the same color with alpha
    if args.std:
        c_succ = line_succ.get_color()
        c_fail = line_fail.get_color()
        # Mean ± std bands (lighter via alpha)
        plt.fill_between(
            t,
            [m - s if (m==m and s==s) else float("nan") for m, s in zip(succ_mean, succ_std)],
            [m + s if (m==m and s==s) else float("nan") for m, s in zip(succ_mean, succ_std)],
            color=c_succ, alpha=0.20, linewidth=0
        )
        plt.fill_between(
            t,
            [m - s if (m==m and s==s) else float("nan") for m, s in zip(fail_mean, fail_std)],
            [m + s if (m==m and s==s) else float("nan") for m, s in zip(fail_mean, fail_std)],
            color=c_fail, alpha=0.20, linewidth=0
        )

    plt.title(args.title, fontsize=25)
    plt.xlabel("Timestep", fontsize=18)
    plt.ylabel("Swap Ratio", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, linewidth=0.6, alpha=0.5)
    plt.legend(fontsize=14)
    plt.tight_layout()
    if args.save:
        plt.savefig(os.path.join(args.save, args.suite_name+'_timewise.png'), dpi=300)
    plt.show()

if __name__ == "__main__":
    main()