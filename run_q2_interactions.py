"""Question 2: compare within-area pairwise spike-train interactions.

The primary interaction is the absolute Pearson correlation between 100 ms
unit spike counts minus its expectation under independent circular shifts.
Every inferential observation is one mouse/area summary, never one unit pair.

Examples
--------
Run synthetic and binning checks:
    python run_q2_interactions.py --self-test

Run the complete primary analysis and robustness checks:
    python run_q2_interactions.py --all

Run only selected mice:
    python run_q2_interactions.py --mice 1 2 3
"""

from __future__ import annotations

import argparse
import gzip
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
from scipy import stats


VALID_MICE = tuple(range(1, 12)) + tuple(range(13, 19))
AREA_IDS = (1, 2, 3)
AREA_LABELS = {area: f"Brain area {area}" for area in AREA_IDS}


@dataclass(frozen=True)
class Config:
    bin_s: float = 0.100
    min_unit_rate_hz: float = 0.05
    n_surrogates: int = 100
    sensitivity_surrogates: int = 20
    min_shift_fraction: float = 0.10
    max_shift_fraction: float = 0.90
    bootstrap_samples: int = 10_000
    equalization_repeats: int = 500
    alpha: float = 0.05
    random_seed: int = 20260705
    fs_lfp: float = 500.0


def load_mouse_spikes(
    mouse: int, data_dir: Path, cfg: Config
) -> tuple[nap.TsGroup, float, pd.DataFrame]:
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
    if len(np.unique(unit_ids)) != len(unit_ids):
        raise ValueError(f"Duplicate unit IDs for Mouse {mouse}.")

    lfp = np.load(folder / "lfp_1.npy", mmap_mode="r")
    recording_duration = float(lfp.shape[1] / cfg.fs_lfp)
    del lfp
    # Restrict to complete bins so Pynapple and the NumPy reference have the
    # same unambiguous right boundary.
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
    qc = pd.DataFrame(
        {
            "mouse": mouse,
            "unit_id": unit_ids,
            "brain_area": unit_areas,
            "firing_rate_hz": [len(spike_by_unit[int(uid)]) / duration for uid in unit_ids],
        }
    )
    return group, duration, qc


def eligible_group(group: nap.TsGroup, cfg: Config) -> nap.TsGroup:
    """Retain units meeting the prespecified whole-recording rate threshold."""
    keep = group.index[np.asarray(group.rates) >= cfg.min_unit_rate_hz]
    return group[list(keep)]


def count_matrix(
    group: nap.TsGroup,
    bin_s: float,
    ep: nap.IntervalSet | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return unit-by-bin Pynapple counts and their unit IDs."""
    counts = group.count(bin_size=bin_s, ep=ep, time_units="s", dtype=np.uint16)
    matrix = np.asarray(counts.values, dtype=np.float64).T
    return matrix, np.asarray(counts.columns, dtype=np.int64)


def numpy_count_reference(
    group: nap.TsGroup, duration: float, bin_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Small transparent reference implementation used only for validation."""
    n_bins = int(round(duration / bin_s))
    edges = np.arange(n_bins + 1, dtype=np.float64) * bin_s
    unit_ids = np.asarray(group.index, dtype=np.int64)
    out = np.zeros((len(unit_ids), n_bins), dtype=np.float64)
    for row, unit_id in enumerate(unit_ids):
        out[row] = np.histogram(np.asarray(group[int(unit_id)].t), bins=edges)[0]
    return out, unit_ids


def standardized_rows(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardize unit count rows and flag rows with nonzero variance."""
    centered = counts - counts.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    valid = np.isfinite(norms) & (norms > np.finfo(float).eps)
    standardized = np.zeros_like(centered, dtype=np.float64)
    standardized[valid] = centered[valid] / norms[valid, None]
    return standardized, valid


def pair_index_table(
    unit_ids: np.ndarray, unit_areas: np.ndarray, valid: np.ndarray
) -> pd.DataFrame:
    """Build indices for all valid within-area unit pairs."""
    rows = []
    for area in AREA_IDS:
        positions = np.flatnonzero((unit_areas == area) & valid)
        for left, right in combinations(positions, 2):
            rows.append(
                {
                    "brain_area": area,
                    "row_i": int(left),
                    "row_j": int(right),
                    "unit_i": int(unit_ids[left]),
                    "unit_j": int(unit_ids[right]),
                }
            )
    return pd.DataFrame(rows)


def pair_interactions(
    counts: np.ndarray,
    unit_ids: np.ndarray,
    unit_areas: np.ndarray,
    cfg: Config,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Compute observed and surrogate-corrected correlations for every pair."""
    standardized, valid = standardized_rows(counts)
    pairs = pair_index_table(unit_ids, unit_areas, valid)
    if pairs.empty:
        return pairs.assign(
            signed_r=pd.Series(dtype=float),
            absolute_r=pd.Series(dtype=float),
            null_absolute_r=pd.Series(dtype=float),
            corrected_absolute_r=pd.Series(dtype=float),
        )

    left = pairs["row_i"].to_numpy()
    right = pairs["row_j"].to_numpy()
    # Matrix multiplication produces only a compact unit-by-unit correlation
    # matrix. Avoid materializing pair-by-time arrays, which can exceed a
    # gigabyte for the larger recordings.
    observed_matrix = standardized @ standardized.T
    observed = observed_matrix[left, right]
    null_sum = np.zeros(len(pairs), dtype=np.float64)
    n_bins = counts.shape[1]
    shift_low = max(1, int(math.ceil(cfg.min_shift_fraction * n_bins)))
    shift_high = min(n_bins, int(math.floor(cfg.max_shift_fraction * n_bins)) + 1)
    if shift_high <= shift_low:
        raise ValueError("Recording is too short for the requested surrogate shifts.")

    shifted = np.empty_like(standardized)
    for _ in range(cfg.n_surrogates):
        shifts = rng.integers(shift_low, shift_high, size=len(unit_ids))
        for row, shift in enumerate(shifts):
            shifted[row] = np.roll(standardized[row], int(shift))
        surrogate_matrix = shifted @ shifted.T
        null_sum += np.abs(surrogate_matrix[left, right])

    null_abs = null_sum / cfg.n_surrogates
    pairs["signed_r"] = observed
    pairs["absolute_r"] = np.abs(observed)
    pairs["null_absolute_r"] = null_abs
    pairs["corrected_absolute_r"] = np.abs(observed) - null_abs
    return pairs


def areas_for_units(group: nap.TsGroup, unit_ids: np.ndarray) -> np.ndarray:
    metadata = group.metadata
    return metadata.loc[unit_ids, "brain_area"].to_numpy(dtype=np.int8)


def summarize_mouse_pairs(
    mouse: int,
    pairs: pd.DataFrame,
    eligible: nap.TsGroup,
    scope: str,
    bin_s: float,
) -> pd.DataFrame:
    rows = []
    metadata = eligible.metadata
    for area in AREA_IDS:
        area_pairs = pairs[pairs["brain_area"] == area]
        n_units = int((metadata["brain_area"] == area).sum())
        rows.append(
            {
                "mouse": mouse,
                "scope": scope,
                "bin_ms": int(round(bin_s * 1000)),
                "brain_area": area,
                "n_units": n_units,
                "n_pairs": int(len(area_pairs)),
                "mean_corrected_absolute_r": area_pairs[
                    "corrected_absolute_r"
                ].mean(),
                "mean_absolute_r": area_pairs["absolute_r"].mean(),
                "mean_signed_r": area_pairs["signed_r"].mean(),
                "mean_null_absolute_r": area_pairs["null_absolute_r"].mean(),
            }
        )
    return pd.DataFrame(rows)


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
    bin_s: float | None = None,
    scope: str = "whole_recording",
    validate_counts: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    analysis_bin = cfg.bin_s if bin_s is None else bin_s
    local_cfg = replace(cfg, bin_s=analysis_bin)
    group, duration, qc = load_mouse_spikes(mouse, data_dir, local_cfg)
    group = eligible_group(group, local_cfg)
    ep = None
    if scope == "task_trials":
        ep = trial_epochs(mouse, data_dir, duration)
    elif scope != "whole_recording":
        raise ValueError(f"Unknown scope: {scope}")

    counts, unit_ids = count_matrix(group, analysis_bin, ep=ep)
    if validate_counts and scope == "whole_recording":
        reference, reference_ids = numpy_count_reference(group, duration, analysis_bin)
        if not np.array_equal(unit_ids, reference_ids) or not np.array_equal(
            counts, reference
        ):
            raise AssertionError(f"Pynapple/NumPy count mismatch for Mouse {mouse}.")

    unit_areas = areas_for_units(group, unit_ids)
    seed_offset = int(mouse * 10_000 + round(analysis_bin * 1e6))
    if scope == "task_trials":
        seed_offset += 5_000_000
    rng = np.random.default_rng(cfg.random_seed + seed_offset)
    pairs = pair_interactions(counts, unit_ids, unit_areas, local_cfg, rng)
    pairs.insert(0, "mouse", mouse)
    pairs.insert(1, "scope", scope)
    pairs.insert(2, "bin_ms", int(round(analysis_bin * 1000)))
    mouse_summary = summarize_mouse_pairs(
        mouse, pairs, group, scope=scope, bin_s=analysis_bin
    )
    qc["eligible"] = qc["unit_id"].isin(group.index)
    return pairs, mouse_summary, qc


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
    mouse_results: pd.DataFrame, metric: str, alpha: float
) -> dict[str, object]:
    wide = mouse_results.pivot(index="mouse", columns="brain_area", values=metric)
    wide = wide.dropna(subset=list(AREA_IDS))
    friedman = stats.friedmanchisquare(*(wide[area] for area in AREA_IDS))
    pairs: list[dict[str, object]] = []
    raw_p = []
    for left, right in combinations(AREA_IDS, 2):
        delta = wide[left] - wide[right]
        test = stats.wilcoxon(wide[left], wide[right], alternative="two-sided")
        raw_p.append(float(test.pvalue))
        pairs.append(
            {
                "left_area": left,
                "right_area": right,
                "mean_difference": float(delta.mean()),
                "median_difference": float(delta.median()),
                "left_wins": int((delta > 0).sum()),
                "right_wins": int((delta < 0).sum()),
                "p_raw": float(test.pvalue),
            }
        )
    for row, adjusted in zip(pairs, holm_adjust(raw_p)):
        row["p_holm"] = adjusted

    means = wide.mean()
    top_area = int(means.idxmax())
    top_tests = [
        row
        for row in pairs
        if top_area in (row["left_area"], row["right_area"])
    ]
    top_beats_both = len(top_tests) == 2
    for row in top_tests:
        other = (
            row["right_area"]
            if row["left_area"] == top_area
            else row["left_area"]
        )
        correct_direction = means[top_area] > means[other]
        top_beats_both &= bool(row["p_holm"] < alpha and correct_direction)
    answer = (
        AREA_LABELS[top_area]
        if top_beats_both
        else "Not enough data / no significant differences"
    )
    return {
        "metric": metric,
        "n_mice": int(len(wide)),
        "friedman_statistic": float(friedman.statistic),
        "friedman_p": float(friedman.pvalue),
        "pairwise": pairs,
        "largest_mean_area": top_area,
        "top_area_beats_both_after_holm": bool(top_beats_both),
        "multiple_choice_answer": answer,
    }


def bootstrap_summary(
    mouse_results: pd.DataFrame, metric: str, cfg: Config
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_seed)
    rows = []
    for area in AREA_IDS:
        values = mouse_results.loc[
            mouse_results["brain_area"] == area, metric
        ].dropna().to_numpy()
        indices = rng.integers(0, len(values), size=(cfg.bootstrap_samples, len(values)))
        boot = values[indices].mean(axis=1)
        rows.append(
            {
                "brain_area": area,
                "candidate": AREA_LABELS[area],
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "ci95_low": float(np.percentile(boot, 2.5)),
                "ci95_high": float(np.percentile(boot, 97.5)),
                "n_mice": int(len(values)),
                "metric": metric,
            }
        )
    return pd.DataFrame(rows)


def equalize_unit_counts(
    primary_pairs: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """Repeatedly subsample each area to the mouse-specific minimum unit count."""
    rng = np.random.default_rng(cfg.random_seed + 9_000_000)
    rows = []
    for mouse, mouse_pairs in primary_pairs.groupby("mouse"):
        units_by_area = {}
        arrays_by_area = {}
        for area in AREA_IDS:
            area_pairs = mouse_pairs[mouse_pairs["brain_area"] == area]
            pair_i = area_pairs["unit_i"].to_numpy(dtype=int)
            pair_j = area_pairs["unit_j"].to_numpy(dtype=int)
            values = area_pairs["corrected_absolute_r"].to_numpy(dtype=float)
            units_by_area[area] = np.unique(np.concatenate([pair_i, pair_j]))
            arrays_by_area[area] = (
                pair_i,
                pair_j,
                values,
            )
        n_equal = min(len(units_by_area[area]) for area in AREA_IDS)
        repeat_values = {area: [] for area in AREA_IDS}
        for _ in range(cfg.equalization_repeats):
            for area in AREA_IDS:
                selected = rng.choice(units_by_area[area], n_equal, replace=False)
                pair_i, pair_j, values = arrays_by_area[area]
                use = np.isin(pair_i, selected) & np.isin(pair_j, selected)
                repeat_values[area].append(values[use].mean())
        for area in AREA_IDS:
            values = np.asarray(repeat_values[area])
            rows.append(
                {
                    "mouse": int(mouse),
                    "brain_area": area,
                    "n_units_equalized": int(n_equal),
                    "equalization_repeats": cfg.equalization_repeats,
                    "mean_corrected_absolute_r": float(np.nanmean(values)),
                    "equalization_sd": float(np.nanstd(values, ddof=1)),
                }
            )
    return pd.DataFrame(rows)


def save_pair_table(frame: pd.DataFrame, path: Path) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)


def plot_paired_mice(
    mouse_results: pd.DataFrame, summary: pd.DataFrame, output_path: Path
) -> None:
    metric = "mean_corrected_absolute_r"
    fig, ax = plt.subplots(figsize=(8, 5.5))
    wide = mouse_results.pivot(index="mouse", columns="brain_area", values=metric)
    for mouse, values in wide.iterrows():
        ax.plot(AREA_IDS, values, color="0.72", linewidth=0.9, alpha=0.8)
        ax.scatter(AREA_IDS, values, color="0.45", s=14, zorder=2)
    ordered = summary.set_index("brain_area").loc[list(AREA_IDS)]
    means = ordered["mean"].to_numpy()
    yerr = np.vstack(
        [
            means - ordered["ci95_low"].to_numpy(),
            ordered["ci95_high"].to_numpy() - means,
        ]
    )
    ax.errorbar(
        AREA_IDS,
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
    ax.set_xticks(AREA_IDS, [AREA_LABELS[area] for area in AREA_IDS])
    ax.set_ylabel("Surrogate-corrected mean |r|")
    ax.set_title("100 ms within-area pairwise interactions across 17 mice")
    ax.axhline(0, color="black", linewidth=0.7, alpha=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_pair_distributions(primary_pairs: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    data = [
        primary_pairs.loc[
            primary_pairs["brain_area"] == area, "corrected_absolute_r"
        ].dropna()
        for area in AREA_IDS
    ]
    parts = ax.violinplot(data, positions=AREA_IDS, showmedians=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("#4393c3")
        body.set_alpha(0.55)
    parts["cmedians"].set_color("#b2182b")
    ax.set_xticks(AREA_IDS, [AREA_LABELS[area] for area in AREA_IDS])
    ax.set_ylabel("Pairwise corrected |r|")
    ax.set_title("Unit-pair distributions (descriptive only)")
    ax.axhline(0, color="black", linewidth=0.7, alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(
    summary: pd.DataFrame,
    tests: dict[str, object],
    equalized_tests: dict[str, object],
    sensitivity: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        "# Question 2 — pairwise spike-train interactions",
        "",
        f"## Answer: {tests['multiple_choice_answer']}",
        "",
        "Spike times are loaded directly from each provided `spikes.npy` file and",
        "assigned to units using `clusters.npy`; no spike detection is performed.",
        "Pynapple `TsGroup.count` creates the binned count trains.",
        "",
        "The primary analysis uses absolute Pearson correlations between 100 ms",
        "unit-count trains, corrected by 100 independent circular-shift surrogates.",
        "Each mouse contributes one value per area.",
        "",
        summary.to_markdown(index=False),
        "",
        f"Friedman p-value: `{tests['friedman_p']:.6g}`.",
        "",
        "Holm-corrected paired Wilcoxon comparisons:",
        "",
        pd.DataFrame(tests["pairwise"]).to_markdown(index=False),
        "",
        "The winning rule requires one area to exceed both alternatives after",
        "Holm correction. A significant omnibus test alone is not sufficient.",
        "",
        "## Unit-count equalization",
        "",
        f"Equalized-analysis answer: **{equalized_tests['multiple_choice_answer']}**.",
        "",
        "## Sensitivity analyses",
        "",
        "These exploratory checks use 20 deterministic surrogates each; the",
        "answer-defining primary analysis uses 100.",
        "",
        sensitivity.to_markdown(index=False),
        "",
        "Pair-level distributions are descriptive; inferential tests use mice.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_self_tests(cfg: Config) -> None:
    assert nap.__version__ == "0.11.3", nap.__version__

    support = nap.IntervalSet(start=0.0, end=2.0)
    group = nap.TsGroup(
        {
            10: nap.Ts(t=np.array([0.02, 0.11, 0.19, 1.99])),
            20: nap.Ts(t=np.array([0.04, 0.15, 1.01])),
        },
        time_support=support,
        metadata={"brain_area": [1, 1]},
    )
    pynapple_counts, pynapple_ids = count_matrix(group, 0.1)
    numpy_counts, numpy_ids = numpy_count_reference(group, 2.0, 0.1)
    assert np.array_equal(pynapple_ids, numpy_ids)
    assert np.array_equal(pynapple_counts, numpy_counts)

    rng = np.random.default_rng(123)
    independent = rng.poisson(0.12, size=(12, 40_000)).astype(float)
    ids = np.arange(12)
    areas = np.ones(12, dtype=int)
    test_cfg = replace(cfg, n_surrogates=30)
    independent_pairs = pair_interactions(
        independent, ids, areas, test_cfg, np.random.default_rng(456)
    )
    assert abs(independent_pairs["corrected_absolute_r"].mean()) < 0.01

    latent = rng.poisson(0.08, size=40_000)
    coupled = rng.poisson(0.05, size=(12, 40_000)).astype(float)
    coupled[:6] += latent
    coupled_pairs = pair_interactions(
        coupled, ids, areas, test_cfg, np.random.default_rng(789)
    )
    within_coupled = coupled_pairs.query("unit_i < 6 and unit_j < 6")
    assert within_coupled["corrected_absolute_r"].mean() > 0.15

    print("Pynapple count parity and synthetic interaction tests passed.")


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
        "pairs": output_dir / "primary_pair_results.csv.gz",
        "mice": output_dir / "primary_mouse_results.csv",
        "qc": output_dir / "unit_qc.csv",
    }
    if reuse_primary and all(path.exists() for path in cached_paths.values()):
        print("Reusing cached primary 100 ms results", flush=True)
        pair_frame = pd.read_csv(cached_paths["pairs"])
        mouse_frame = pd.read_csv(cached_paths["mice"])
        qc_frame = pd.read_csv(cached_paths["qc"])
        cached_mice = set(mouse_frame["mouse"].unique())
        requested_mice = set(mice)
        if cached_mice != requested_mice:
            raise ValueError(
                f"Cached mice {sorted(cached_mice)} do not match requested mice "
                f"{sorted(requested_mice)}."
            )
    else:
        primary_pairs = []
        primary_mice = []
        unit_qc = []
        for index, mouse in enumerate(mice):
            print(f"Primary 100 ms analysis: Mouse {mouse}", flush=True)
            pairs, mouse_summary, qc = analyze_mouse(
                mouse,
                data_dir,
                cfg,
                validate_counts=(index == 0),
            )
            primary_pairs.append(pairs)
            primary_mice.append(mouse_summary)
            unit_qc.append(qc)
        pair_frame = pd.concat(primary_pairs, ignore_index=True)
        mouse_frame = pd.concat(primary_mice, ignore_index=True)
        qc_frame = pd.concat(unit_qc, ignore_index=True)
    metric = "mean_corrected_absolute_r"
    summary = bootstrap_summary(mouse_frame, metric, cfg)
    tests = paired_statistics(mouse_frame, metric, cfg.alpha)

    print("Unit-count equalization", flush=True)
    equalized = equalize_unit_counts(pair_frame, cfg)
    equalized_tests = paired_statistics(equalized, metric, cfg.alpha)

    sensitivity_columns = [
        "scope",
        "bin_ms",
        "area_1_mean",
        "area_2_mean",
        "area_3_mean",
        "friedman_p",
        "largest_mean_area",
        "multiple_choice_answer",
        "n_surrogates",
    ]
    sensitivity_rows = []
    if include_sensitivity:
        sensitivity_cfg = replace(cfg, n_surrogates=cfg.sensitivity_surrogates)
        sensitivity_specs = [
            (0.050, "whole_recording"),
            (0.200, "whole_recording"),
            (0.100, "task_trials"),
        ]
        for bin_s, scope in sensitivity_specs:
            frames = []
            for mouse in mice:
                print(
                    f"Sensitivity {scope}, {int(bin_s * 1000)} ms: Mouse {mouse}",
                    flush=True,
                )
                _, mouse_summary, _ = analyze_mouse(
                    mouse, data_dir, sensitivity_cfg, bin_s=bin_s, scope=scope
                )
                frames.append(mouse_summary)
            sensitivity_frame = pd.concat(frames, ignore_index=True)
            sensitivity_test = paired_statistics(sensitivity_frame, metric, cfg.alpha)
            means = sensitivity_frame.groupby("brain_area")[metric].mean()
            sensitivity_rows.append(
                {
                    "scope": scope,
                    "bin_ms": int(bin_s * 1000),
                    "area_1_mean": means[1],
                    "area_2_mean": means[2],
                    "area_3_mean": means[3],
                    "friedman_p": sensitivity_test["friedman_p"],
                    "largest_mean_area": sensitivity_test["largest_mean_area"],
                    "multiple_choice_answer": sensitivity_test[
                        "multiple_choice_answer"
                    ],
                    "n_surrogates": cfg.sensitivity_surrogates,
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows, columns=sensitivity_columns)

    save_pair_table(pair_frame, output_dir / "primary_pair_results.csv.gz")
    mouse_frame.to_csv(output_dir / "primary_mouse_results.csv", index=False)
    qc_frame.to_csv(output_dir / "unit_qc.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    equalized.to_csv(output_dir / "unit_equalized_results.csv", index=False)
    sensitivity.to_csv(output_dir / "sensitivity_results.csv", index=False)
    (output_dir / "statistical_tests.json").write_text(
        json.dumps(
            {
                "primary": tests,
                "unit_equalized": equalized_tests,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "config.json").write_text(
        json.dumps(asdict(cfg), indent=2), encoding="utf-8"
    )
    plot_paired_mice(
        mouse_frame, summary, output_dir / "paired_mouse_interactions.png"
    )
    plot_pair_distributions(pair_frame, output_dir / "pair_distributions.png")
    write_report(
        summary,
        tests,
        equalized_tests,
        sensitivity,
        output_dir / "report.md",
    )
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
        "--output-dir", type=Path, default=Path("results/q2_interactions")
    )
    parser.add_argument(
        "--surrogates",
        type=int,
        default=Config.n_surrogates,
        help="Number of circular-shift surrogates.",
    )
    parser.add_argument(
        "--sensitivity-surrogates",
        type=int,
        default=Config.sensitivity_surrogates,
        help="Surrogates for exploratory 50/200 ms and trial-only checks.",
    )
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="Skip 50/200 ms and task-trial sensitivity analyses.",
    )
    parser.add_argument(
        "--reuse-primary",
        action="store_true",
        help="Reuse matching primary outputs and run only remaining summaries/checks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = replace(
        Config(),
        n_surrogates=args.surrogates,
        sensitivity_surrogates=args.sensitivity_surrogates,
    )
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
