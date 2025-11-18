import argparse, json, os, re, glob
import math
import numpy as np
import matplotlib.pyplot as plt 

MP4_PATTERN = re.compile(r"episode=(\d+).*?success=(True|False)", re.IGNORECASE)
EP_FROM_JSON_PATTERN = re.compile(r"ep_(\d+)_(\d+)", re.IGNORECASE)

def build_episode_success_map(rollout_dir: str):
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
        raise RuntimeError(f"No success signal of any episode under {rollout_dir}.s")
    return ep2success

def iter_suite_jsons(results_dir: str, suite: int):
    pattern = os.path.join(results_dir, f"ep_{suite}_*_two_metrics.json")
    files = sorted(glob.glob(pattern))
    if not files:
        pattern2 = os.path.join(results_dir, f"ep_{suite}_*.json")
        files = sorted(glob.glob(pattern2))
    return files

def parse_suite_episode_from_json(json_path: str):
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
        raise ValueError(f"Can not analyse from {json_path}.")
    if data is None:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    return int(m2.group(1)), int(m2.group(2)), data

def collect_segment_swapratio10(json_data: dict):
    rows = json_data.get("rows", [])
    if len(rows) < 10:
        raise ValueError(f"rows less than 10：len={len(rows)}")
    return [float(rows[i]["swapratio_mean"]) for i in range(10)]

def avg_list(vals, cnt):
    return [v / cnt for v in vals] if cnt > 0 else [float("nan")] * 10

def add_lists(a, b):
    return [x + y for x, y in zip(a, b)]

def zeros10():
    return [0.0] * 10

def add_lists_sumsq(a, b):
    return [x + y for x, y in zip(a, b)]

def square_list(a):
    return [x * x for x in a]

def std_from_sum_sumsq_means(n, sumsq, means):
    if n <= 1:
        return [float("nan")] * len(means)
    out = []
    for ss, m in zip(sumsq, means):
        var = (ss - n * (m ** 2)) / (n - 1)
        out.append(math.sqrt(var) if var > 0 else 0.0)
    return out

def process_one_suite(suite, results_dir, ep2success, episodes_per_suite):
    json_files = iter_suite_jsons(results_dir, suite)
    if not json_files:
        return None

    succ_sum = zeros10(); succ_cnt = 0
    fail_sum = zeros10(); fail_cnt = 0
    succ_sumsq = zeros10()
    fail_sumsq = zeros10()

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

        vals = collect_segment_swapratio10(data)
        if succ_flag:
            succ_sum = add_lists(succ_sum, vals)
            succ_sumsq = add_lists_sumsq(succ_sumsq, square_list(vals))
            succ_cnt += 1
        else:
            fail_sum = add_lists(fail_sum, vals)
            fail_sumsq = add_lists_sumsq(fail_sumsq, square_list(vals))
            fail_cnt += 1
        used += 1

    succ_avg = avg_list(succ_sum, succ_cnt)
    fail_avg = avg_list(fail_sum, fail_cnt)

    return {
        "suite": suite,
        "succ_avg": succ_avg, "succ_cnt": succ_cnt, "succ_sum": succ_sum, "succ_sumsq": succ_sumsq,
        "fail_avg": fail_avg, "fail_cnt": fail_cnt, "fail_sum": fail_sum, "fail_sumsq": fail_sumsq,
        "used": used
    }

def _plot_overall_with_std_errorbar(
    succ_mean, succ_std, fail_mean, fail_std, title: str, save_path: str
):
    def _pad(a, n, fill=np.nan):
        a = list(a)
        if len(a) < n: a += [fill] * (n - len(a))
        return np.array(a, dtype=float)

    xs_len = max(len(succ_mean), len(fail_mean), 10)
    xs = np.arange(1, xs_len + 1)

    y_s = _pad(succ_mean, xs_len, np.nan)
    e_s = _pad(succ_std,  xs_len, 0.0)
    y_f = _pad(fail_mean, xs_len, np.nan)
    e_f = _pad(fail_std,  xs_len, 0.0)

    plt.figure(figsize=(8.8, 5.0))
    plt.errorbar(xs, y_s, yerr=e_s, fmt="-o", capsize=3, label="Success")
    plt.errorbar(xs, y_f, yerr=e_f, fmt="--s", capsize=3, label="Failure")

    plt.title(title, fontsize=25)
    plt.xlabel("Segment", fontsize=18)
    plt.ylabel("Swap Ratio", fontsize=18)
    plt.xticks(xs, fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, alpha=0.5, linewidth=0.6)
    plt.legend(fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="None")
    parser.add_argument("--suite", type=str, required=True)
    parser.add_argument("--suite_name", type=str, default='libero_spatial')
    parser.add_argument("--sd", type=str, default='7')  
    parser.add_argument("--results_base", type=str, default="../results")
    parser.add_argument("--rollouts_base", type=str, default="../rollouts")
    parser.add_argument("--episodes_per_suite", type=int, default=50)
    parser.add_argument("--num_suites", type=int, default=10)
    parser.add_argument("--std", action="store_true")
    parser.add_argument("--save_path", type=str, default="../plots/swp")
    args = parser.parse_args()
    
    ep2success = build_episode_success_map(os.path.join(args.rollouts_base, args.suite_name, 'seed_'+args.sd))

    suites = []
    if args.suite.lower() == "all" if isinstance(args.suite, str) else False or str(args.suite) == "0":
        suites = list(range(1, args.num_suites + 1))
    else:
        suites = [int(args.suite)]

    tot_succ_sum = zeros10(); tot_succ_cnt = 0
    tot_fail_sum = zeros10(); tot_fail_cnt = 0
    tot_succ_sumsq = zeros10()
    tot_fail_sumsq = zeros10()

    simple_succ_sum = zeros10(); simple_succ_k = 0
    simple_fail_sum = zeros10(); simple_fail_k = 0

    any_output = False
    results_dir = os.path.join(args.results_base, args.suite_name, 'seed_'+args.sd)

    for s in suites:
        res = process_one_suite(s, results_dir, ep2success, args.episodes_per_suite)
        if res is None:
            print(f"No suite {s}'s JSON file.")
            continue

        any_output = True
        print(f"Success (episodes={res['succ_cnt']}): [{', '.join(f'{x:.6f}' if x==x else 'nan' for x in res['succ_avg'])}]")
        print(f"Failure (episodes={res['fail_cnt']}): [{', '.join(f'{x:.6f}' if x==x else 'nan' for x in res['fail_avg'])}]")
        if res["used"] < args.episodes_per_suite:
            print(f"[INFO] suite {s}: {res['used']}/{args.episodes_per_suite}.")

        tot_succ_sum = add_lists(tot_succ_sum, res["succ_sum"])
        tot_succ_sumsq = add_lists_sumsq(tot_succ_sumsq, res["succ_sumsq"])
        tot_succ_cnt += res["succ_cnt"]

        tot_fail_sum = add_lists(tot_fail_sum, res["fail_sum"])
        tot_fail_sumsq = add_lists_sumsq(tot_fail_sumsq, res["fail_sumsq"])
        tot_fail_cnt += res["fail_cnt"]

        if res["succ_cnt"] > 0:
            simple_succ_sum = add_lists(simple_succ_sum, res["succ_avg"])
            simple_succ_k += 1
        if res["fail_cnt"] > 0:
            simple_fail_sum = add_lists(simple_fail_sum, res["fail_avg"])
            simple_fail_k += 1

    if not any_output:
        raise FileNotFoundError("Not found")

    overall_succ_avg = avg_list(tot_succ_sum, tot_succ_cnt)
    overall_fail_avg = avg_list(tot_fail_sum, tot_fail_cnt)
    print(f"Success overall (episodes={tot_succ_cnt}): [{', '.join(f'{x:.6f}' if x==x else 'nan' for x in overall_succ_avg)}]")
    print(f"Failure overall (episodes={tot_fail_cnt}): [{', '.join(f'{x:.6f}' if x==x else 'nan' for x in overall_fail_avg)}]")

    simple_succ_avg = avg_list(simple_succ_sum, simple_succ_k)
    simple_fail_avg = avg_list(simple_fail_sum, simple_fail_k)
    print(f"Success mean-of-suites (suites={simple_succ_k}): [{', '.join(f'{x:.6f}' if x==x else 'nan' for x in simple_succ_avg)}]")
    print(f"Failure mean-of-suites (suites={simple_fail_k}): [{', '.join(f'{x:.6f}' if x==x else 'nan' for x in simple_fail_avg)}]")

    succ_std_overall = std_from_sum_sumsq_means(tot_succ_cnt, tot_succ_sumsq, overall_succ_avg) if args.std else None
    fail_std_overall = std_from_sum_sumsq_means(tot_fail_cnt, tot_fail_sumsq, overall_fail_avg) if args.std else None

    if args.std:
        title = args.suite_name
        _plot_overall_with_std_errorbar(
            overall_succ_avg, succ_std_overall or [],
            overall_fail_avg, fail_std_overall or [],
            title=title, save_path=os.path.join(args.save_path, args.suite_name+'_segment.png')
        )
    else:
        seg_idx = list(range(1, 11))
        plt.figure(figsize=(8.8, 5.0))
        line1, = plt.plot(seg_idx, overall_succ_avg, "-",  linewidth=2.6, label=f"Success")
        line2, = plt.plot(seg_idx, overall_fail_avg, "--", linewidth=2.6, label=f"Failure")
        plt.title(args.suite_name, fontsize=25)
        plt.xlabel("Segment", fontsize=18)
        plt.ylabel("Swap Ratio", fontsize=18)
        plt.xticks(seg_idx, fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True, linewidth=0.6, alpha=0.5)
        plt.legend(fontsize=14)
        plt.tight_layout()
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        plt.savefig(os.path.join(args.save_path, args.suite_name+'_segment.png'), dpi=300)
        plt.show()
        plt.close()

if __name__ == "__main__":
    main()
