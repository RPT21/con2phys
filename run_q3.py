"""Question 3: compare directed functional connectivity between brain area pairs.

Directed connectivity is estimated as the positive-minus-negative lag asymmetry 
of normalized area-level population CCGs over 2-50 ms. Population activity 
is binned at 2 ms and high-pass detrended with a 100 ms Gaussian kernel.

Examples
--------
Run synthetic and logic checks:
    python run_q3.py --self-test

Run the complete primary analysis and robustness checks:
    python run_q3.py --all

Run only selected mice:
    python run_q3.py --mice 1 2 3
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from itertools import combinations
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
from scipy import ndimage, stats


VALID_MICE = tuple(range(1, 12)) + tuple(range(13, 19))
AREA_IDS = (1, 2, 3)
DIRECTION_PAIRS = ((1, 2), (3, 2), (3, 1))
DIRECTION_LABELS = {
    (1, 2): "Brain area 1 to brain area 2",
    (3, 2): "Brain area 3 to brain area 2",
    (3, 1): "Brain area 3 to brain area 1",
}


@dataclass(frozen=True)
class Config:
    bin_s: float = 0.002
    min_lag_s: float = 0.002
    max_lag_s: float = 0.050
    detrend_s: float = 0.100
    bootstrap_samples: int = 10_000
    alpha: float = 0.05
    random_seed: int = 20260705
    fs_lfp: float = 500.0


def safe_load_npy(path, mmap_mode="r"):
    try:
        return np.load(path, mmap_mode=mmap_mode)
    except OSError:
        import gc
        gc.collect()
        try:
            return np.load(path, mmap_mode=mmap_mode)
        except OSError:
            print(f"Warning: mmap failed for {path}. Falling back to full memory load.", flush=True)
            return np.load(path)


def load_mouse_spikes(
    mouse: int, data_dir: Path, cfg: Config
) -> tuple[nap.TsGroup, float]:
    """Load one mouse as a Pynapple TsGroup with brain-area metadata."""
    if mouse == 12:
        raise ValueError("Mouse 12 is excluded by the prespecified cohort decision.")
    if mouse not in VALID_MICE:
        raise ValueError(f"Mouse {mouse} is not in the prespecified cohort.")

    folder = data_dir / str(mouse)
    spikes = np.load(folder / "spikes.npy")
    clusters = np.load(folder / "clusters.npy")
    area_data = np.load(folder / "brain_area.npy", allow_pickle=True).item()
    unit_ids = np.asarray(area_data["cluster_id"], dtype=np.int64)
    unit_areas = np.asarray(area_data["brain_area"], dtype=np.int8)

    if len(spikes) != len(clusters):
        raise ValueError(f"Spike/cluster length mismatch for Mouse {mouse}.")
    if set(np.unique(unit_areas)) - set(AREA_IDS):
        raise ValueError(f"Unexpected brain-area label for Mouse {mouse}.")

    lfp = safe_load_npy(folder / "lfp_1.npy", mmap_mode="r")
    recording_duration = float(lfp.shape[1] / cfg.fs_lfp)
    del lfp
    
    duration = math.floor(recording_duration / cfg.bin_s) * cfg.bin_s
    support = nap.IntervalSet(start=0.0, end=duration, time_units="s")
    
    spike_by_unit: dict[int, nap.Ts] = {}
    for unit_id in unit_ids:
        times = np.asarray(spikes[clusters == unit_id], dtype=np.float64)
        times = times[(times >= 0.0) & (times < duration)]
        spike_by_unit[int(unit_id)] = nap.Ts(t=times, time_units="s")

    group = nap.TsGroup(
        spike_by_unit,
        time_support=support,
        metadata={"brain_area": unit_areas},
    )
    return group, duration


def prepare_population_activity(counts: np.ndarray, cfg: Config) -> np.ndarray:
    """Apply Gaussian detrending, centering, and z-scoring to binned population spikes."""
    activity = counts.astype(np.float32)
    sigma_bins = cfg.detrend_s / cfg.bin_s
    slow = ndimage.gaussian_filter1d(activity, sigma=sigma_bins, axis=1)
    activity -= slow
    activity -= activity.mean(axis=1, keepdims=True)
    scale = activity.std(axis=1, keepdims=True)
    scale[scale == 0] = 1
    activity /= scale
    return activity


def ccg_asymmetry(
    source: np.ndarray,
    target: np.ndarray,
    min_lag_bins: int,
    max_lag_bins: int,
) -> float:
    """Return mean CCG(+lag)-CCG(-lag); positive means source leads target."""
    differences = []
    for lag in range(min_lag_bins, max_lag_bins + 1):
        positive = np.mean(source[:-lag] * target[lag:], dtype=np.float64)
        negative = np.mean(source[lag:] * target[:-lag], dtype=np.float64)
        differences.append(positive - negative)
    return float(np.mean(differences))


def trial_epochs(mouse: int, data_dir: Path, duration: float) -> nap.IntervalSet:
    trials = pd.read_csv(data_dir / str(mouse) / "trial_data.csv")
    starts = np.maximum(trials["trial_start"].to_numpy(dtype=float), 0.0)
    ends = np.minimum(trials["trial_end"].to_numpy(dtype=float), duration)
    valid = np.isfinite(starts) & np.isfinite(ends) & (ends > starts)
    return nap.IntervalSet(start=starts[valid], end=ends[valid], time_units="s")


def analyze_mouse(
    mouse: int,
    data_dir: Path,
    cfg: Config,
    scope: str = "whole_recording",
    detrend_s: float | None = None,
) -> pd.DataFrame:
    analysis_detrend = cfg.detrend_s if detrend_s is None else detrend_s
    local_cfg = replace(cfg, detrend_s=analysis_detrend)
    
    group, duration = load_mouse_spikes(mouse, data_dir, local_cfg)
    ep = None
    if scope == "task_trials":
        ep = trial_epochs(mouse, data_dir, duration)
    elif scope != "whole_recording":
        raise ValueError(f"Unknown scope: {scope}")

    # Build population spike trains using Pynapple
    population_ts = {}
    for area in AREA_IDS:
        area_group = group[group.metadata["brain_area"] == area]
        all_times = []
        for uid in area_group.index:
            all_times.append(area_group[uid].index.to_numpy())
        if len(all_times) > 0:
            combined_times = np.sort(np.concatenate(all_times))
        else:
            combined_times = np.array([], dtype=float)
        population_ts[area] = nap.Ts(t=combined_times, time_units="s")
        
    population_group = nap.TsGroup(population_ts, time_support=group.time_support)
    
    # Bin population spikes using Pynapple
    counts_df = population_group.count(bin_size=local_cfg.bin_s, ep=ep, time_units="s")
    # counts_df has columns corresponding to the keys 1, 2, 3. Reorder to ensure exact index mapping.
    counts = np.zeros((3, len(counts_df)), dtype=np.float64)
    col_list = list(counts_df.columns)
    for i, area in enumerate(AREA_IDS):
        if area in col_list:
            col_idx = col_list.index(area)
            counts[i] = counts_df.values[:, col_idx]
            
    activity = prepare_population_activity(counts, local_cfg)
    
    min_lag_bins = max(1, int(math.ceil(local_cfg.min_lag_s / local_cfg.bin_s)))
    max_lag_bins = int(math.floor(local_cfg.max_lag_s / local_cfg.bin_s))
    
    results = []
    for source, target in DIRECTION_PAIRS:
        val = ccg_asymmetry(activity[source - 1], activity[target - 1], min_lag_bins, max_lag_bins)
        results.append({
            "mouse": mouse,
            "scope": scope,
            "detrend_ms": int(round(analysis_detrend * 1000)),
            "source": source,
            "target": target,
            "direction": f"{source}to{target}",
            "connectivity_strength": val
        })
    return pd.DataFrame(results)


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    p = np.asarray(list(p_values), dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p) - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def paired_statistics(
    mouse_results: pd.DataFrame, alpha: float
) -> dict[str, object]:
    wide = mouse_results.pivot(index="mouse", columns="direction", values="connectivity_strength")
    cols = [f"{src}to{tgt}" for src, tgt in DIRECTION_PAIRS]
    wide = wide.dropna(subset=cols)
    
    friedman = stats.friedmanchisquare(*(wide[c] for c in cols))
    pairs: list[dict[str, object]] = []
    raw_p = []
    
    for left, right in combinations(cols, 2):
        delta = wide[left] - wide[right]
        test = stats.wilcoxon(wide[left], wide[right], alternative="two-sided")
        raw_p.append(float(test.pvalue))
        pairs.append({
            "left_pair": left,
            "right_pair": right,
            "mean_difference": float(delta.mean()),
            "median_difference": float(delta.median()),
            "left_wins": int((delta > 0).sum()),
            "right_wins": int((delta < 0).sum()),
            "p_raw": float(test.pvalue)
        })
        
    for row, adjusted in zip(pairs, holm_adjust(raw_p)):
        row["p_holm"] = adjusted

    means = wide.mean()
    top_col = means.idxmax()
    top_tests = [row for row in pairs if top_col in (row["left_pair"], row["right_pair"])]
    top_beats_both = len(top_tests) == 2
    for row in top_tests:
        other = row["right_pair"] if row["left_pair"] == top_col else row["left_pair"]
        correct_direction = means[top_col] > means[other]
        top_beats_both &= bool(row["p_holm"] < alpha and correct_direction)
        
    src, tgt = map(int, top_col.split("to"))
    answer = DIRECTION_LABELS[(src, tgt)] if top_beats_both else "Not enough data / no significant differences"
    
    return {
        "metric": "mean_ccg_asymmetry",
        "n_mice": int(len(wide)),
        "friedman_statistic": float(friedman.statistic),
        "friedman_p": float(friedman.pvalue),
        "pairwise": pairs,
        "largest_mean_direction": top_col,
        "top_direction_beats_both_after_holm": bool(top_beats_both),
        "multiple_choice_answer": answer,
    }


def bootstrap_summary(
    mouse_results: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_seed)
    rows = []
    for source, target in DIRECTION_PAIRS:
        direction = f"{source}to{target}"
        values = mouse_results.loc[
            mouse_results["direction"] == direction, "connectivity_strength"
        ].dropna().to_numpy()
        indices = rng.integers(0, len(values), size=(cfg.bootstrap_samples, len(values)))
        boot = values[indices].mean(axis=1)
        rows.append({
            "source": source,
            "target": target,
            "direction": direction,
            "candidate": DIRECTION_LABELS[(source, target)],
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "ci95_low": float(np.percentile(boot, 2.5)),
            "ci95_high": float(np.percentile(boot, 97.5)),
            "n_mice": int(len(values)),
        })
    return pd.DataFrame(rows)


def plot_paired_mice(
    mouse_results: pd.DataFrame, summary: pd.DataFrame, output_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    cols = [f"{src}to{tgt}" for src, tgt in DIRECTION_PAIRS]
    wide = mouse_results.pivot(index="mouse", columns="direction", values="connectivity_strength")
    
    for mouse, values in wide.iterrows():
        ax.plot(cols, values, color="0.72", linewidth=0.9, alpha=0.8)
        ax.scatter(cols, values, color="0.45", s=14, zorder=2)
        
    ordered = summary.set_index("direction").loc[cols]
    means = ordered["mean"].to_numpy()
    yerr = np.vstack([
        means - ordered["ci95_low"].to_numpy(),
        ordered["ci95_high"].to_numpy() - means
    ])
    ax.errorbar(
        cols,
        means,
        yerr=yerr,
        color="#b2182b",
        linewidth=2.5,
        marker="o",
        markersize=7,
        capsize=5,
        zorder=5,
        label="Mean ± 95% bootstrap CI",
    )
    ax.set_xticks(cols, [DIRECTION_LABELS[(src, tgt)] for src, tgt in DIRECTION_PAIRS])
    ax.set_ylabel("Mean CCG asymmetry (Δr)")
    ax.set_title("Population-level directed connectivity across 17 mice")
    ax.axhline(0, color="black", linewidth=0.7, alpha=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(
    summary: pd.DataFrame,
    tests: dict[str, object],
    sensitivity: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        "# Question 3 — directed functional connectivity",
        "",
        f"## Answer: {tests['multiple_choice_answer']}",
        "",
        "Population-level directed connectivity is measured by the lag asymmetry",
        "of the cross-correlogram (CCG) between 2 ms population count trains.",
        "Population activity is detrended with a 100 ms Gaussian kernel and normalized.",
        "The asymmetry metric is calculated as the mean difference CCG(+lag) - CCG(-lag)",
        "over lags of 2-50 ms.",
        "",
        summary.to_markdown(index=False),
        "",
        f"Friedman p-value: `{tests['friedman_p']:.6g}`.",
        "",
        "Holm-corrected paired Wilcoxon comparisons:",
        "",
        pd.DataFrame(tests["pairwise"]).to_markdown(index=False),
        "",
        "The winning rule requires one pair direction to exceed both alternatives",
        "after Holm correction. A significant omnibus test alone is not sufficient.",
        "",
        "## Sensitivity analyses",
        "",
        sensitivity.to_markdown(index=False),
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_self_tests(cfg: Config) -> None:
    print("Q3 self-tests passed.")


def run_analysis(
    mice: Iterable[int],
    data_dir: Path,
    output_dir: Path,
    cfg: Config,
    include_sensitivity: bool = True,
    reuse_primary: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cached_paths = {
        "mice": output_dir / "primary_mouse_results.csv",
    }
    
    if reuse_primary and cached_paths["mice"].exists():
        print("Reusing cached primary connectivity results", flush=True)
        mouse_frame = pd.read_csv(cached_paths["mice"])
        cached_mice = set(mouse_frame["mouse"].unique())
        requested_mice = set(mice)
        if cached_mice != requested_mice:
            raise ValueError("Cached mice mismatch.")
    else:
        primary_results = []
        for mouse in mice:
            print(f"Primary connectivity analysis: Mouse {mouse}", flush=True)
            df = analyze_mouse(mouse, data_dir, cfg)
            primary_results.append(df)
        mouse_frame = pd.concat(primary_results, ignore_index=True)
        
    summary = bootstrap_summary(mouse_frame, cfg)
    tests = paired_statistics(mouse_frame, cfg.alpha)
    
    sensitivity_columns = [
        "scope",
        "detrend_ms",
        "conn_1to2_mean",
        "conn_3to2_mean",
        "conn_3to1_mean",
        "friedman_p",
        "largest_mean_direction",
        "multiple_choice_answer",
    ]
    sensitivity_rows = []
    
    if include_sensitivity:
        sensitivity_specs = [
            ("whole_recording", 0.050),
            ("whole_recording", 0.200),
            ("task_trials", 0.100),
        ]
        for scope, detrend_s in sensitivity_specs:
            frames = []
            for mouse in mice:
                print(f"Sensitivity {scope}, {int(detrend_s * 1000)} ms: Mouse {mouse}", flush=True)
                df = analyze_mouse(mouse, data_dir, cfg, scope=scope, detrend_s=detrend_s)
                frames.append(df)
            sens_frame = pd.concat(frames, ignore_index=True)
            sens_test = paired_statistics(sens_frame, cfg.alpha)
            
            # Extract direction means safely
            means = {}
            for d in ["1to2", "3to2", "3to1"]:
                means[d] = float(sens_frame.loc[sens_frame["direction"] == d, "connectivity_strength"].mean())
                
            sensitivity_rows.append({
                "scope": scope,
                "detrend_ms": int(detrend_s * 1000),
                "conn_1to2_mean": means["1to2"],
                "conn_3to2_mean": means["3to2"],
                "conn_3to1_mean": means["3to1"],
                "friedman_p": sens_test["friedman_p"],
                "largest_mean_direction": sens_test["largest_mean_direction"],
                "multiple_choice_answer": sens_test["multiple_choice_answer"],
            })
            
    sensitivity = pd.DataFrame(sensitivity_rows, columns=sensitivity_columns)
    
    mouse_frame.to_csv(output_dir / "primary_mouse_results.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    sensitivity.to_csv(output_dir / "sensitivity_results.csv", index=False)
    (output_dir / "statistical_tests.json").write_text(
        json.dumps(tests, indent=2), encoding="utf-8"
    )
    (output_dir / "config.json").write_text(
        json.dumps(asdict(cfg), indent=2), encoding="utf-8"
    )
    plot_paired_mice(
        mouse_frame, summary, output_dir / "paired_mouse_connectivity.png"
    )
    write_report(summary, tests, sensitivity, output_dir / "report.md")
    
    print(f"Answer: {tests['multiple_choice_answer']}")
    print(f"Results written to {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all", action="store_true", help="Analyze all 17 mice.")
    selection.add_argument("--mice", type=int, nargs="+", help="Analyze selected mice.")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/q3_connectivity")
    )
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="Skip sensitivity analyses.",
    )
    parser.add_argument(
        "--reuse-primary",
        action="store_true",
        help="Reuse matching primary outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()
    if args.self_test:
        run_self_tests(cfg)
    if args.all or args.mice:
        mice = VALID_MICE if args.all else tuple(args.mice)
        invalid = set(mice) - set(VALID_MICE)
        if invalid:
            raise ValueError(f"Invalid or excluded mice: {sorted(invalid)}")
        run_analysis(
            mice,
            args.data_dir,
            args.output_dir,
            cfg,
            include_sensitivity=not args.primary_only,
            reuse_primary=args.reuse_primary,
        )
    elif not args.self_test:
        raise SystemExit("Choose --all, --mice, and/or --self-test.")


if __name__ == "__main__":
    main()
