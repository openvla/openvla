import argparse, json, os, re, glob, math
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

MP4_PATTERN = re.compile(r"episode=(\d+).*?success=(True|False)", re.IGNORECASE)
EP_FROM_JSON_PATTERN = re.compile(r"ep_(\d+)_(\d+)", re.IGNORECASE)

FOCUS_KEY = "focus_shift_seg_mean_map"
P_KEYS = ("p1", "p3", "p5", "p10")

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

def iter_suite_jsons(results_dir: str, suite: int) -> List[str]:
    pattern = os.path.join(results_dir, f"ep_{suite}_*_two_metrics.json")
    files = sorted(glob.glob(pattern))
    if not files:
        pattern2 = os.path.join(results_dir, f"ep_{suite}_*.json")
        files = sorted(glob.glob(pattern2))
    return files

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

# ---- helpers ----
def _ensure_len_f(a: List[float], n: int):
    if len(a) < n:
        a += [0.0] * (n - len(a))

def _ensure_len_i(a: List[int], n: int):
    if len(a) < n:
        a += [0] * (n - len(a))

def add_episode_lists(sum_list: List[float], cnt_list: List[int], sumsq_list: List[float], vals: List[float]):
    _ensure_len_f(sum_list,  len(vals))
    _ensure_len_i(cnt_list,  len(vals))
    _ensure_len_f(sumsq_list, len(vals))
    for i, v in enumerate(vals):
        try:
            fv = float(v)
        except Exception:
            continue
        if math.isnan(fv):
            continue
        sum_list[i]   += fv
        sumsq_list[i] += fv * fv
        cnt_list[i]   += 1

def mean_std_from_sum_cnt_sumsq(sum_list: List[float], cnt_list: List[int], sumsq_list: List[float]) -> Tuple[List[float], List[float]]:
    means, stds = [], []
    for s, c, ssq in zip(sum_list, cnt_list, sumsq_list):
        if c <= 0:
            means.append(float("nan")); stds.append(float("nan")); continue
        m = s / c
        if c > 1:
            var_num = ssq - c * (m * m)
            var = max(0.0, var_num) / (c - 1)
            std = math.sqrt(var)
        else:
            std = 0.0
        means.append(float(m))
        stds.append(float(std))
    return means, stds

def div_lists(sum_list: List[float], cnt_list: List[int]) -> List[float]:
    out = []
    for s, c in zip(sum_list, cnt_list):
        out.append(float("nan") if c <= 0 else (s / c))
    return out

def extract_focus_lists(json_data: Dict[str, Any]) -> Dict[str, List[float]]:
    out = {k: [] for k in P_KEYS}
    fs = json_data.get(FOCUS_KEY, None)
    if isinstance(fs, dict):
        for k in P_KEYS:
            v = fs.get(k, [])
            if isinstance(v, list):
                cleaned = []
                for x in v:
                    try:
                        cleaned.append(float(x))
                    except Exception:
                        cleaned.append(float("nan"))
                out[k] = cleaned
    return out

def format_vec(vec: List[float]) -> str:
    return "[" + ", ".join(("nan" if (v != v) else f"{v:.6f}") for v in vec) + "]"

def process_one_suite(suite: int, results_dir: str, ep2success: Dict[int, bool], episodes_per_suite: int):
    json_files = iter_suite_jsons(results_dir, suite)
    if not json_files:
        return None

    succ_sum = {k: [] for k in P_KEYS}; succ_cnt = {k: [] for k in P_KEYS}; succ_sumsq = {k: [] for k in P_KEYS}; succ_eps = 0
    fail_sum = {k: [] for k in P_KEYS}; fail_cnt = {k: [] for k in P_KEYS}; fail_sumsq = {k: [] for k in P_KEYS}; fail_eps = 0
    used = 0
    missing_map = 0

    for jp in json_files:
        suite_i, ep_in_suite, data = parse_suite_episode_from_json(jp)
        if suite_i != suite:
            continue
        global_ep = (suite_i - 1) * episodes_per_suite + ep_in_suite
        succ_flag = ep2success.get(global_ep)
        if succ_flag is None:
            missing_map += 1
            continue

        lists = extract_focus_lists(data)
        if all((len(lists[k]) == 0) for k in P_KEYS):
            continue

        if succ_flag:
            for k in P_KEYS:
                add_episode_lists(succ_sum[k], succ_cnt[k], succ_sumsq[k], lists[k])
            succ_eps += 1
        else:
            for k in P_KEYS:
                add_episode_lists(fail_sum[k], fail_cnt[k], fail_sumsq[k], lists[k])
            fail_eps += 1
        used += 1

    return {
        "suite": suite,
        "succ_sum": succ_sum, "succ_cnt": succ_cnt, "succ_sumsq": succ_sumsq, "succ_eps": succ_eps,
        "fail_sum": fail_sum, "fail_cnt": fail_cnt, "fail_sumsq": fail_sumsq, "fail_eps": fail_eps,
        "used": used, "missing_map": missing_map
    }

def compute_mean_std_maps(res):
    succ = {k: mean_std_from_sum_cnt_sumsq(res["succ_sum"][k], res["succ_cnt"][k], res["succ_sumsq"][k]) for k in P_KEYS}
    fail = {k: mean_std_from_sum_cnt_sumsq(res["fail_sum"][k], res["fail_cnt"][k], res["fail_sumsq"][k]) for k in P_KEYS}
    return succ, fail

def compute_avg_maps(res):
    succ_avg = {k: div_lists(res["succ_sum"][k], res["succ_cnt"][k]) for k in P_KEYS}
    fail_avg = {k: div_lists(res["fail_sum"][k], res["fail_cnt"][k]) for k in P_KEYS}
    return succ_avg, fail_avg

def print_suite(res):
    succ_avg, fail_avg = compute_avg_maps(res)
    print(f"Success (episodes={res['succ_eps']}):")
    for k in P_KEYS:
        print(f"  {k}: {format_vec(succ_avg[k])}")
    print(f"Failure (episodes={res['fail_eps']}):")
    for k in P_KEYS:
        print(f"  {k}: {format_vec(fail_avg[k])}")

# -------------------- Plotting helpers --------------------
def _plot_one_p_with_std(pkey: str, succ_mean: List[float], succ_std: List[float],
                         fail_mean: List[float], fail_std: List[float],
                         title: str, save_path: str):
    xs_len = max(len(succ_mean), len(fail_mean))
    xs = np.arange(xs_len)

    def _pad(a, n, fill=np.nan):
        a = list(a)
        if len(a) < n:
            a += [fill] * (n - len(a))
        return np.array(a, dtype=float)

    y_s = _pad(succ_mean, xs_len, np.nan)
    e_s = _pad(succ_std,  xs_len, 0.0)
    y_f = _pad(fail_mean, xs_len, np.nan)
    e_f = _pad(fail_std,  xs_len, 0.0)

    plt.figure(figsize=(7.4, 3.8))
    plt.errorbar(xs, y_s, yerr=e_s, fmt="-o", capsize=3, label="Success")
    plt.errorbar(xs, y_f, yerr=e_f, fmt="-s", capsize=3, label="Failure")

    plt.xlabel("Segment", fontsize=18)
    plt.ylabel("Focus shift", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title(title, fontsize=25)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()

def plot_suite_figures(res, out_dir: str, prefix: str):
    succ_ms, fail_ms = compute_mean_std_maps(res)
    for p in P_KEYS:
        s_mean, s_std = succ_ms[p]
        f_mean, f_std = fail_ms[p]
        title = f"{prefix} ({p})"
        save_path = os.path.join(out_dir, f"{prefix.replace(' ', '_')}_{p}.png")
        _plot_one_p_with_std(p, s_mean, s_std, f_mean, f_std, title, save_path)

# ----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="None")
    parser.add_argument("--suite", type=str, required=True)
    parser.add_argument("--results_base", type=str, default="../results")
    parser.add_argument("--rollouts_base", type=str, default="../rollouts")
    parser.add_argument("--suite_name", type=str, default="libero_spatial")
    parser.add_argument("--sd", type=str, "7")
    parser.add_argument("--episodes_per_suite", type=int, default=50)
    parser.add_argument("--num_suites", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fig_dir", type=str, default='../plots/shift')
    args = parser.parse_args()

    ep2success = build_episode_success_map(os.path.join(args.rollouts_base, args.suite_name, 'seed_'+args.sd))

    is_all = (isinstance(args.suite, str) and args.suite.lower() == "all") or str(args.suite) == "0"
    suites = list(range(1, args.num_suites + 1)) if is_all else [int(args.suite)]

    os.makedirs(args.fig_dir, exist_ok=True)

    tot_succ_sum = {k: [] for k in P_KEYS}; tot_succ_cnt = {k: [] for k in P_KEYS}; tot_succ_sumsq = {k: [] for k in P_KEYS}
    tot_fail_sum = {k: [] for k in P_KEYS}; tot_fail_cnt = {k: [] for k in P_KEYS}; tot_fail_sumsq = {k: [] for k in P_KEYS}

    any_output = False
    for s in suites:
        res = process_one_suite(s, os.path.join(args.results_base, args.suite_name, 'seed_'+args.sd), ep2success, args.episodes_per_suite)
        if res is None:
            print(f"[WARN] No suite {s}'s json file.")
            continue
        any_output = True

        if not is_all or args.verbose:
            print_suite(res)
            if res["used"] < args.episodes_per_suite:
                print(f"[INFO] suite {s}:  {res['used']}/{args.episodes_per_suite} missing {res['missing_map']}.")
            suite_prefix = f"suite_{s}"
            out_dir_suite = os.path.join(args.fig_dir, f"suite_{s}")
            plot_suite_figures(res, out_dir_suite, suite_prefix)

        for k in P_KEYS:
            _ensure_len_f(tot_succ_sum[k],   len(res["succ_sum"][k]))
            _ensure_len_i(tot_succ_cnt[k],   len(res["succ_cnt"][k]))
            _ensure_len_f(tot_succ_sumsq[k], len(res["succ_sumsq"][k]))
            _ensure_len_f(tot_fail_sum[k],   len(res["fail_sum"][k]))
            _ensure_len_i(tot_fail_cnt[k],   len(res["fail_cnt"][k]))
            _ensure_len_f(tot_fail_sumsq[k], len(res["fail_sumsq"][k]))
            for i, ssum in enumerate(res["succ_sum"][k]):
                tot_succ_sum[k][i]   += ssum
                tot_succ_cnt[k][i]   += res["succ_cnt"][k][i]
                tot_succ_sumsq[k][i] += res["succ_sumsq"][k][i]
            for i, fsum in enumerate(res["fail_sum"][k]):
                tot_fail_sum[k][i]   += fsum
                tot_fail_cnt[k][i]   += res["fail_cnt"][k][i]
                tot_fail_sumsq[k][i] += res["fail_sumsq"][k][i]

    if not any_output:
        raise FileNotFoundError("Statistically Unsuccessful.")

    overall_res = {
        "suite": 0,
        "succ_sum": tot_succ_sum, "succ_cnt": tot_succ_cnt, "succ_sumsq": tot_succ_sumsq, "succ_eps": 0,
        "fail_sum": tot_fail_sum, "fail_cnt": tot_fail_cnt, "fail_sumsq": tot_fail_sumsq, "fail_eps": 0,
    }
    succ_ms_overall, fail_ms_overall = compute_mean_std_maps(overall_res)

    title = "\n=== Overall (across selected suites & episodes) — per-segment mean ± std ==="
    print(title)
    def _simple_means(ms_map):  # {p: ([mean],[std])} -> {p: [mean]}
        return {k: v[0] for k, v in ms_map.items()}
    overall_means_s = _simple_means(succ_ms_overall)
    overall_means_f = _simple_means(fail_ms_overall)
    print("Success (mean):")
    for k in P_KEYS:
        print(f"  {k}: {format_vec(overall_means_s[k])}")
    print("Failure (mean):")
    for k in P_KEYS:
        print(f"  {k}: {format_vec(overall_means_f[k])}")

    overall_prefix = args.suite_name+'_segment'
    os.makedirs(args.fig_dir, exist_ok=True)
    for p in P_KEYS:
        s_mean, s_std = succ_ms_overall[p]
        f_mean, f_std = fail_ms_overall[p]
        save_path = os.path.join(args.fig_dir, f"{overall_prefix}_{p}.png")
        _plot_one_p_with_std(p, s_mean, s_std, f_mean, f_std,
                             f"Libero Long ({p})",
                             save_path)

if __name__ == "__main__":
    main()
