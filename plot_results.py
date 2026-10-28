#!/usr/bin/env python3
"""
plot_results.py — robust plotting for epymarl-llm / sacred results.

Usage examples
--------------
# Plot everything found under results/sacred (default behaviour)
python plot_results.py

# Only plot specific maps/environments
python plot_results.py --envs 2o_20b_vs_2r 5z_vs_1ul

# Only plot specific algorithms
python plot_results.py --algs LMAC QMIX

# Only plot specific metrics
python plot_results.py --metrics test_battle_won_mean test_return_mean

# Flag whether LMAC beats a win-rate threshold at the end of training
# (a lightweight "did the reproduction work?" check)
python plot_results.py --success-threshold 70

# Point at a different results directory / output directory
python plot_results.py --base-dir results/sacred --out-dir plots

Key fixes vs. the previous version
-----------------------------------
1. Smoothing bias: the old EMA started "warm" from the very first raw value,
   which systematically drags the first ~1/(1-weight) points toward that
   single noisy sample. This version uses TensorBoard-style *debiased* EMA,
   which divides out that initial-condition bias.
2. Never silently overwrites a plot: every image is versioned
   (name.png -> name_v2.png -> name_v3.png ...) so you can always tell the
   newest run apart from older ones instead of clobbering history.
3. Filtering: --envs / --algs / --metrics let you regenerate just the
   plot(s) you care about instead of re-walking + re-plotting everything.
4. Robustness: corrupt/partial metrics.json or config.json no longer kill
   the whole script — they're logged and skipped, and you get a summary of
   exactly what was skipped and why.
5. Informativeness: per-config seed counts, detection of missing/duplicate
   seeds, min/max/mean/std of final performance, and (optionally) a simple
   pass/fail reproduction check against a target threshold.
"""

import argparse
import json
import os
import sys
import traceback
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("default")
sns.set_theme(style="whitegrid")

DEFAULT_COLOR_MAP = {"LMAC": "#eb4c34", "QMIX": "#4c72b0"}
FALLBACK_PALETTE = sns.color_palette("tab10")


# --------------------------------------------------------------------------- #
# Smoothing
# --------------------------------------------------------------------------- #
def smooth_curve_debiased(values, weight=0.85):
    """
    TensorBoard-style debiased exponential moving average.

    The naive EMA (last = last*w + (1-w)*x, seeded with last=values[0])
    is biased early on because it starts "warm" at a single raw sample.
    Debiasing divides out that initial-condition bias:

        s_t = w * s_(t-1) + (1 - w) * x_t         (s_0 = 0)
        smoothed_t = s_t / (1 - w^(t+1))

    This keeps the same eventual smoothing strength but removes the
    systematic pull toward the first data point in early timesteps.
    """
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return values
    smoothed = np.zeros_like(values)
    running = 0.0
    for t, x in enumerate(values):
        running = weight * running + (1 - weight) * x
        debias = 1 - weight ** (t + 1)
        smoothed[t] = running / debias if debias > 1e-12 else x
    return smoothed


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def find_all_runs(base_dir, skip_log):
    """
    Walk base_dir for sacred run folders (must contain metrics.json).
    Returns a list of dicts: {alg, env, seed, run_dir, metrics}.
    Anything unreadable/corrupt is recorded in skip_log instead of raising.
    """
    runs = []
    if not os.path.isdir(base_dir):
        skip_log.append(f"[FATAL] base_dir does not exist: {base_dir}")
        return runs

    for root, _dirs, files in os.walk(base_dir):
        if "metrics.json" not in files:
            continue

        metrics_path = os.path.join(root, "metrics.json")
        config_path = os.path.join(root, "config.json")

        path_parts = os.path.normpath(root).split(os.sep)
        alg_fallback = path_parts[-3] if len(path_parts) >= 3 else "Unknown"
        env_fallback = path_parts[-2] if len(path_parts) >= 2 else "Unknown"
        seed_fallback = path_parts[-1] if len(path_parts) >= 1 else "unknown"

        alg_name, env_name, seed = alg_fallback, env_fallback, seed_fallback

        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                alg_name = str(config.get("name", alg_fallback))
                env_name = str(config.get("env_args", {}).get("map_name", env_fallback))
                seed = str(config.get("seed", seed_fallback))
            except (json.JSONDecodeError, OSError) as e:
                skip_log.append(f"[WARN] Could not parse config.json at {config_path}: {e} "
                                 f"(falling back to path-derived names)")

        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            skip_log.append(f"[SKIP] Unreadable/corrupt metrics.json at {metrics_path}: {e}")
            continue

        if not metrics:
            skip_log.append(f"[SKIP] Empty metrics.json at {metrics_path}")
            continue

        runs.append({
            "alg": alg_name.upper(),
            "env": env_name,
            "seed": seed,
            "run_dir": root,
            "metrics": metrics,
        })

    return runs


# --------------------------------------------------------------------------- #
# Filename versioning
# --------------------------------------------------------------------------- #
def versioned_path(directory, stem, ext=".png"):
    """
    Return a path that never overwrites an existing file.
    First save: stem.png
    Next save (same stem already exists): stem_v2.png, stem_v3.png, ...
    The highest version number is always the most recent plot.
    """
    candidate = os.path.join(directory, f"{stem}{ext}")
    if not os.path.exists(candidate):
        return candidate
    v = 2
    while True:
        candidate = os.path.join(directory, f"{stem}_v{v}{ext}")
        if not os.path.exists(candidate):
            return candidate
        v += 1


# --------------------------------------------------------------------------- #
# Main plotting routine
# --------------------------------------------------------------------------- #
def generate_plots(base_dir="results/sacred",
                    out_dir="plots",
                    envs_filter=None,
                    algs_filter=None,
                    metrics_filter=None,
                    smooth_weight=0.85,
                    expected_seeds=None,
                    success_threshold=None,
                    final_window=10,
                    min_seeds=1):
    """
    envs_filter / algs_filter / metrics_filter: None = include everything,
    otherwise a set of allowed (case-insensitive) names.
    expected_seeds: e.g. [1,2,3,4,5] — used to report missing seeds per config.
    success_threshold: if set, flags PASS/FAIL on the final metric value for
    any metric containing "battle_won", per (env, alg).
    """
    skip_log = []
    all_runs = find_all_runs(base_dir, skip_log)

    if not all_runs:
        print(f"[!] No metrics found under '{base_dir}'.")
        for line in skip_log:
            print("   ", line)
        return

    def norm_set(x):
        return {s.strip().upper() for s in x} if x else None

    envs_filter = norm_set(envs_filter)
    algs_filter = norm_set(algs_filter)
    metrics_filter = norm_set(metrics_filter)

    default_metrics = {"TEST_BATTLE_WON_MEAN"}
    target_metrics = metrics_filter if metrics_filter else default_metrics

    # env -> metric -> alg -> list of (seed, steps, values, run_dir)
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    filtered_out = {"env": 0, "alg": 0, "metric": 0}
    for run in all_runs:
        env, alg, seed = run["env"], run["alg"], run["seed"]

        if envs_filter and env.upper() not in envs_filter:
            filtered_out["env"] += 1
            continue
        if algs_filter and alg.upper() not in algs_filter:
            filtered_out["alg"] += 1
            continue

        matched_any_metric = False
        for metric_name, metric_data in run["metrics"].items():
            if metric_name.upper() not in target_metrics:
                continue
            steps = np.array(metric_data.get("steps", []))
            values = np.array(metric_data.get("values", []))
            if len(steps) == 0 or len(values) == 0 or len(steps) != len(values):
                skip_log.append(f"[SKIP] {run['run_dir']}: metric '{metric_name}' "
                                 f"has mismatched/empty steps vs values, skipping this metric.")
                continue
            data[env][metric_name][alg].append((seed, steps, values, run["run_dir"]))
            matched_any_metric = True

        if not matched_any_metric:
            filtered_out["metric"] += 1

    # Stitch: a resumed/continued experiment creates a NEW sacred run each time
    # (results/sacred/.../<run_id>/metrics.json), and each one only contains the
    # datapoints logged during THAT session — not the full history. So for a seed
    # resumed multiple times, "most points in one run" is NOT the most complete
    # curve; it's usually the *original* truncated run, since later continuation
    # sessions each only add a handful of new points on top. The correct fix is
    # to merge every sacred run for a given seed into one continuous series,
    # keyed by step, rather than picking a single "best" run and discarding rest.
    for env_name, metrics_dict in data.items():
        for metric_name, algs_dict in metrics_dict.items():
            for alg_name, runs in algs_dict.items():
                seed_groups = defaultdict(list)
                for seed, steps, values, run_dir in runs:
                    seed_groups[seed].append((steps, values, run_dir))

                merged_runs = []
                for seed, group in seed_groups.items():
                    if len(group) == 1:
                        steps, values, run_dir = group[0]
                        merged_runs.append((seed, steps, values, run_dir))
                        continue

                    # Sort each partial run by its own starting step, so that when
                    # two runs report the same step (e.g. the checkpoint step a
                    # continuation resumed from), the later-chronological run's
                    # value wins rather than being arbitrarily overwritten.
                    group_sorted = sorted(
                        group, key=lambda g: g[0][0] if len(g[0]) else 0
                    )
                    merged = {}
                    for steps, values, _run_dir in group_sorted:
                        for s, v in zip(steps, values):
                            merged[s] = v

                    merged_steps = np.array(sorted(merged.keys()))
                    merged_values = np.array([merged[s] for s in merged_steps])
                    run_dirs = ", ".join(rd for _s, _v, rd in group)
                    merged_runs.append((seed, merged_steps, merged_values,
                                         f"merged({len(group)} runs)"))
                    skip_log.append(
                        f"[MERGE] {env_name}/{alg_name}/{metric_name} seed {seed}: "
                        f"stitched {len(group)} sacred runs into one continuous "
                        f"series ({len(merged_steps)} points, up to step "
                        f"{int(merged_steps[-1]) if len(merged_steps) else 0}). "
                        f"Source runs: {run_dirs}"
                    )

                algs_dict[alg_name] = merged_runs

    if not any(data.values()):
        print("[!] Filters excluded every run — nothing to plot.")
        print(f"    Excluded by env filter: {filtered_out['env']}, "
              f"by alg filter: {filtered_out['alg']}, "
              f"by metric filter/mismatch: {filtered_out['metric']}")
        return

    os.makedirs(out_dir, exist_ok=True)

    print("=" * 100)
    print(f"{'ENVIRONMENT':<18} | {'ALGORITHM':<8} | {'METRIC':<24} | {'SEEDS':<18} | "
          f"{'FINAL (mean ± std)':<22} | {'RANGE (min–max)'}")
    print("=" * 100)

    reproduction_report = []

    for env_name, metrics_dict in sorted(data.items()):
        for metric_name, algs_dict in sorted(metrics_dict.items()):
            if not algs_dict:
                continue

            plt.figure(figsize=(9, 5.5))
            has_plotted = False
            color_i = 0

            for alg_name, runs in sorted(algs_dict.items()):
                if not runs:
                    continue

                if len(runs) < min_seeds:
                    skip_log.append(
                        f"[SKIP-PLOT] {env_name}/{alg_name}/{metric_name}: only {len(runs)} "
                        f"seed(s) present (< --min-seeds {min_seeds}), not plotting this curve yet."
                    )
                    continue

                seeds_used = [r[0] for r in runs]
                dupes = [s for s in set(seeds_used) if seeds_used.count(s) > 1]
                if dupes:
                    skip_log.append(f"[WARN] {env_name}/{alg_name}/{metric_name}: "
                                     f"duplicate seed(s) {dupes} found — both copies "
                                     f"are included in the mean, which will understate variance.")

                if expected_seeds:
                    missing = sorted(set(str(s) for s in expected_seeds) - set(seeds_used))
                    if missing:
                        skip_log.append(f"[WARN] {env_name}/{alg_name}/{metric_name}: "
                                         f"missing expected seed(s) {missing}")

                # Align on the run with the most steps recorded; interpolate the rest onto it.
                longest_run = max(runs, key=lambda r: len(r[1]))
                common_steps = longest_run[1]

                interpolated_values, final_values = [], []
                for seed, steps, values, run_dir in runs:
                    if len(steps) < 2:
                        skip_log.append(f"[SKIP] {run_dir}: only {len(steps)} datapoint(s) "
                                         f"for '{metric_name}', too few to plot.")
                        continue
                    interp_val = np.interp(common_steps, steps, values)
                    smoothed_val = smooth_curve_debiased(interp_val, weight=smooth_weight)
                    interpolated_values.append(smoothed_val)
                    window = values[-min(final_window, len(values)):]
                    final_values.append(np.mean(window))

                if not interpolated_values:
                    continue

                interpolated_values = np.array(interpolated_values)
                mean_values = np.mean(interpolated_values, axis=0)
                std_values = np.std(interpolated_values, axis=0)

                is_win_rate = "battle_won" in metric_name.lower()
                scale = 100.0 if (is_win_rate and np.max(mean_values) <= 1.0) else 1.0

                mean_plot = mean_values * scale
                std_plot = std_values * scale
                final_mean = float(np.mean(final_values)) * scale
                final_std = float(np.std(final_values)) * scale
                final_min = float(np.min(final_values)) * scale
                final_max = float(np.max(final_values)) * scale

                unit = "%" if is_win_rate else ""
                seed_str = ",".join(sorted(seeds_used, key=lambda s: (len(s), s)))
                print(f"{env_name:<18} | {alg_name:<8} | {metric_name:<24} | "
                      f"{seed_str:<18} | {final_mean:6.2f} ± {final_std:4.2f}{unit:<3} "
                      f"| [{final_min:6.2f}, {final_max:6.2f}]{unit}  ({len(runs)} seeds)")

                if success_threshold is not None and is_win_rate:
                    verdict = "PASS" if final_mean >= success_threshold else "FAIL"
                    reproduction_report.append(
                        (env_name, alg_name, metric_name, final_mean, success_threshold, verdict)
                    )

                c = DEFAULT_COLOR_MAP.get(alg_name.upper())
                if c is None:
                    c = FALLBACK_PALETTE[color_i % len(FALLBACK_PALETTE)]
                    color_i += 1

                plt.plot(common_steps, mean_plot, label=f"{alg_name} ({len(runs)} seeds)",
                          linewidth=2, color=c)
                plt.fill_between(common_steps, mean_plot - std_plot, mean_plot + std_plot,
                                  alpha=0.18, color=c)
                has_plotted = True

            if has_plotted:
                y_label = "Test Win Rate (%)" if "battle_won" in metric_name.lower() \
                    else metric_name.replace("_", " ").title()
                plt.title(f"{env_name} — {y_label}", fontsize=14, fontweight="bold", pad=12)
                plt.xlabel("Environment Timesteps", fontsize=12)
                plt.ylabel(y_label, fontsize=12)
                if "battle_won" in metric_name.lower():
                    plt.ylim(-5, 105)
                plt.legend(loc="lower right", frameon=True, fontsize=11)
                plt.grid(True, linestyle="--", alpha=0.6)
                plt.tight_layout()

                stem = f"{env_name}_{metric_name}"
                save_path = versioned_path(out_dir, stem)
                plt.savefig(save_path, dpi=300)
                print(f"    -> saved {save_path}")
            plt.close()

    print("=" * 100)

    if filtered_out["env"] or filtered_out["alg"] or filtered_out["metric"]:
        print(f"\n[i] Filtered out: {filtered_out['env']} run(s) by --envs, "
              f"{filtered_out['alg']} run(s) by --algs, "
              f"{filtered_out['metric']} run(s) with no matching --metrics.")

    if skip_log:
        print(f"\n[i] {len(skip_log)} issue(s) encountered (skipped/warned, did not stop the run):")
        for line in skip_log:
            print("   ", line)

    if success_threshold is not None:
        print(f"\n[i] Reproduction check (threshold = {success_threshold}%):")
        if not reproduction_report:
            print("    No win-rate metrics matched the current filters — nothing to check.")
        for env_name, alg_name, metric_name, final_mean, thresh, verdict in reproduction_report:
            print(f"    {env_name:<18} {alg_name:<8} {metric_name:<24} "
                  f"{final_mean:6.2f}% vs {thresh:6.2f}%  -> {verdict}")

    print(f"\n[✓] Done. Plots are in '{out_dir}/'.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Plot sacred results with filtering, robust error handling, "
                    "debiased smoothing, and versioned output filenames."
    )
    p.add_argument("--base-dir", default="results/sacred",
                    help="Root directory to walk for sacred run folders (default: results/sacred)")
    p.add_argument("--out-dir", default="plots",
                    help="Directory to save plots into (default: plots)")
    p.add_argument("--envs", nargs="*", default=None,
                    help="Only plot these map/environment names (default: all found)")
    p.add_argument("--algs", nargs="*", default=None,
                    help="Only plot these algorithm names, e.g. LMAC QMIX (default: all found)")
    p.add_argument("--metrics", nargs="*", default=None,
                    help="Only plot these metric keys, e.g. test_battle_won_mean "
                        "(default: test_battle_won_mean only)")
    p.add_argument("--smooth-weight", type=float, default=0.85,
                    help="EMA smoothing weight in [0,1). Higher = smoother. Default 0.85.")
    p.add_argument("--expected-seeds", nargs="*", default=None,
                    help="Seeds you expect per config, e.g. 1 2 3 4 5. "
                        "Used only to report missing seeds, not to filter data.")
    p.add_argument("--success-threshold", type=float, default=None,
                    help="If set, flags PASS/FAIL against this final win-rate percentage "
                        "for every (env, alg) with a battle_won metric.")
    p.add_argument("--final-window", type=int, default=10,
                    help="Number of trailing raw datapoints averaged for the 'final value' "
                        "reported in the summary table (default: 10).")
    p.add_argument("--min-seeds", type=int, default=1,
                    help="Minimum number of seeds required for an (env, alg, metric) curve "
                        "to be plotted. Curves with fewer seeds are skipped and reported "
                        "instead of silently plotted as partial data. Default: 1 (no floor).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        generate_plots(
            base_dir=args.base_dir,
            out_dir=args.out_dir,
            envs_filter=args.envs,
            algs_filter=args.algs,
            metrics_filter=args.metrics,
            smooth_weight=args.smooth_weight,
            expected_seeds=args.expected_seeds,
            success_threshold=args.success_threshold,
            final_window=args.final_window,
            min_seeds=args.min_seeds,
        )
    except Exception:
        print("[FATAL] Unhandled error while generating plots:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)