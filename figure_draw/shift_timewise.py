# compute_focusshift_timestep_by_suite.py
import argparse, json, os, re, glob, math
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

MP4_PATTERN = re.compile(r"episode=(\d+).*?success=(True|False)", re.IGNORECASE)
EP_FROM_JSON_PATTERN = re.compile(r"ep_(\d+)_(\d+)", re.IGNORECASE)  # ep_{suite}_{episode}

FOCUS_PAIR_KEY = "focus_shift_pair_t_map"
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

def _ensure_len_f(a: List[float], n: int):
    if len(a) < n:
        a += [0.0] * (n - len(a))

def _ensure_len_i(a: List[int], n: int):
    if len(a) < n:
        a += [0] * (n - len(a))

def _ensure_len_agg(agg: Dict[str, Dict[str, List]], n: int):
    _ensure_len_f(agg["sum"], n)
    _ensure_len_f(agg["sum_sq"], n)
    _ensure_len_i(agg["cnt"], n)

def _push_series(agg: Dict[str, Dict[str, List]], series: List[float], clip_len: int):
    if clip_len <= 0:
        return
    s = series[:clip_len]
    _ensure_len_agg(agg, clip_len)
    for i, v in enumerate(s):
        try:
            fv = float(v)
        except Exception:
            continue
        if math.isnan(fv):
            continue
        agg["sum"][i] += fv
        agg["sum_sq"][i] += fv * fv
        agg["cnt"][i] += 1

def _finalize_mean_std(agg: Dict[str, Dict[str, List]]) -> Tuple[np.ndarray, np.ndarray]:
    sum_arr = np.array(agg["sum"], dtype=float)
    sum_sq_arr = np.array(agg["sum_sq"], dtype=float)
    cnt_arr = np.array(agg["cnt"], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sum_arr / np.maximum(1.0, cnt_arr)
        var = (sum_sq_arr / np.maximum(1.0, cnt_arr)) - mean * mean
        var = np.where(var < 0.0, 0.0, var)
        std = np.sqrt(var)
        mean = np.where(cnt_arr <= 0.0, np.nan, mean)
        std  = np.where(cnt_arr <= 0.0, np.nan, std)
    return mean, std

def _get_T_from_json(data: Dict[str, Any], pmap: Dict[str, List[float]]) -> int:
    if isinstance(data.get("T", None), int):
        return int(data["T"])
    lengths = [len(pmap.get(k, [])) for k in P_KEYS]
    lengths = [L for L in lengths if L > 0]
    return min(lengths) if lengths else 0

def extract_focus_pair_map(json_data: Dict[str, Any]) -> Dict[str, List[float]]:
    out = {k: [] for k in P_KEYS}
    mp = json_data.get(FOCUS_PAIR_KEY, None)
    if isinstance(mp, dict):
        for k in P_KEYS:
            v = mp.get(k, [])
            if isinstance(v, list):
                cleaned = []
                for x in v:
                    try:
                        cleaned.append(float(x))
                    except Exception:
                        cleaned.append(float("nan"))
                out[k] = cleaned
    return out

def find_min_T_for_suite(suite: int, results_dir: str, ep2success: Dict[int, bool], episodes_per_suite: int) -> int:
    json_files = iter_suite_jsons(results_dir, suite)
    min_T = None
    for jp in json_files:
        suite_i, ep_in_suite, data = parse_suite_episode_from_json(jp)
        if suite_i != suite:
            continue
        global_ep = (suite_i - 1) * episodes_per_suite + ep_in_suite
        if ep2success.get(global_ep) is None:
            continue
        pmap = extract_focus_pair_map(data)
        if all((len(pmap[k]) == 0) for k in P_KEYS):
            continue
        T = _get_T_from_json(data, pmap)
        if T <= 0:
            continue
        min_T = T if (min_T is None or T < min_T) else min_T
    return int(min_T or 0)

def find_min_T_overall(suites: List[int], results_dir: str, ep2success: Dict[int, bool], episodes_per_suite: int) -> int:
    min_T = None
    for s in suites:
        t = find_min_T_for_suite(s, results_dir, ep2success, episodes_per_suite)
        if t <= 0:
            continue
        min_T = t if (min_T is None or t < min_T) else min_T
    return int(min_T or 0)

def process_one_suite_timestep(
    suite: int, results_dir: str, ep2success: Dict[int, bool], episodes_per_suite: int, min_T: int
):
    json_files = iter_suite_jsons(results_dir, suite)
    if not json_files or min_T <= 0:
        return None

    agg = {
        "success": {k: {"sum": [], "sum_sq": [], "cnt": []} for k in P_KEYS},
        "failure": {k: {"sum": [], "sum_sq": [], "cnt": []} for k in P_KEYS},
    }
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

        pmap = extract_focus_pair_map(data)
        if all((len(pmap[k]) == 0) for k in P_KEYS):
            continue

        label = "success" if succ_flag else "failure"
        for k in P_KEYS:
            _push_series(agg[label][k], pmap[k], clip_len=min_T)

        used += 1

    out = {"suite": suite, "used": used, "missing_map": missing_map, "timestep": {}, "min_T": int(min_T)}
    for label in ("success", "failure"):
        out["timestep"][label] = {}
        for k in P_KEYS:
            mean, std = _finalize_mean_std(agg[label][k])
            out["timestep"][label][k] = {"mean": mean, "std": std}
    return out

def _plot_timestep_one_p(pkey: str,
                         succ_mean: np.ndarray, succ_std: np.ndarray,
                         fail_mean: np.ndarray, fail_std: np.ndarray,
                         title: str, save_path: str):
    xs = np.arange(max(len(succ_mean), len(fail_mean)))
    plt.figure(figsize=(8.0, 3.6))

    if len(succ_mean) > 0:
        sm = succ_mean
        ss = succ_std
        plt.plot(xs[:len(sm)], sm, marker="o", label="Success")
        plt.fill_between(xs[:len(sm)], sm - ss, sm + ss, alpha=0.2)

    if len(fail_mean) > 0:
        fm = fail_mean
        fs = fail_std
        plt.plot(xs[:len(fm)], fm, marker="s", label="Failure")
        plt.fill_between(xs[:len(fm)], fm - fs, fm + fs, alpha=0.2)

    plt.xlabel("Timestep", fontsize=18)
    plt.ylabel("Focus shift", fontsize=18)
    plt.title(title, fontsize=25)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()

def plot_suite_timestep_figures(res_timestep: Dict[str, Any], out_dir: str, prefix: str):
    os.makedirs(out_dir, exist_ok=True)
    for p in P_KEYS:
        succ = res_timestep["timestep"]["success"][p]
        fail = res_timestep["timestep"]["failure"][p]
        title = f"{prefix} | {p} — timestep-wise focus shift (mean ± std)  (min_T={res_timestep['min_T']})"
        save_path = os.path.join(out_dir, f"{prefix.replace(' ', '_')}_{p}_timestep.png")
        _plot_timestep_one_p(
            p,
            np.array(succ["mean"]), np.array(succ["std"]),
            np.array(fail["mean"]), np.array(fail["std"]),
            title, save_path
        )

def main():
    parser = argparse.ArgumentParser(
        description="None"
    )
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

    overall_min_T = find_min_T_overall(suites, os.path.join(args.results_base, args.suite_name, 'seed_'+args.sd), ep2success, args.episodes_per_suite)
    if overall_min_T <= 0:
        raise RuntimeError("No effective T")
    print(f"[INFO] overall min_T = {overall_min_T}")

    overall = {
        "success": {k: {"sum": [], "sum_sq": [], "cnt": []} for k in P_KEYS},
        "failure": {k: {"sum": [], "sum_sq": [], "cnt": []} for k in P_KEYS},
    }

    any_output = False
    for s in suites:
        suite_min_T = find_min_T_for_suite(s, os.path.join(args.results_base, args.suite_name, 'seed_'+args.sd), ep2success, args.episodes_per_suite)
        if suite_min_T <= 0:
            continue

        res_ts = process_one_suite_timestep(
            s, os.path.join(args.results_base, args.suite_name, 'seed_'+args.sd), ep2success, args.episodes_per_suite, min_T=min(overall_min_T, suite_min_T)
        )
        if res_ts is None:
            continue
        any_output = True

        if not is_all or args.verbose:
            suite_prefix = f"suite_{s}"
            out_dir_suite = os.path.join(fig_dir, suite_prefix)
            plot_suite_timestep_figures(res_ts, out_dir_suite, suite_prefix)

        json_files = iter_suite_jsons(os.path.join(args.results_base, args.suite_name, 'seed_'+args.sd), s)
        for jp in json_files:
            suite_i, ep_in_suite, data = parse_suite_episode_from_json(jp)
            if suite_i != s:
                continue
            global_ep = (suite_i - 1) * args.episodes_per_suite + ep_in_suite
            succ_flag = ep2success.get(global_ep)
            if succ_flag is None:
                continue
            label = "success" if succ_flag else "failure"
            pmap = extract_focus_pair_map(data)
            if all((len(pmap[k]) == 0) for k in P_KEYS):
                continue
            for k in P_KEYS:
                _push_series(overall[label][k], pmap[k], clip_len=overall_min_T)

    if not any_output:
        raise FileNotFoundError("Statistically Unsuccessful")

    overall_means_stds = {"success": {}, "failure": {}}
    for label in ("success", "failure"):
        for k in P_KEYS:
            m, sstd = _finalize_mean_std(overall[label][k])
            overall_means_stds[label][k] = (m, sstd)

    overall_prefix = args.suite_name+"_timestep"
     
    for k in P_KEYS:
        succ_m, succ_s = overall_means_stds["success"][k]
        fail_m, fail_s = overall_means_stds["failure"][k]
        title = f"{args.suite_name} ({k})"
        save_path = os.path.join(args.fig_dir, f"{overall_prefix}_{k}.png")
        _plot_timestep_one_p(k, succ_m, succ_s, fail_m, fail_s, title, save_path)

    print(f"Done. Figures saved under: {fig_dir}")

if __name__ == "__main__":
    main()
