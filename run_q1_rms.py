"""Run the selected robust-RMS ripple-density analysis for Question 1."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage, signal, stats


FS = 500.0
VALID_MICE = tuple(range(1, 12)) + tuple(range(13, 19))
AREAS = (1, 2, 3)
RIPPLE_BAND = (100.0, 200.0)
CHANNELS_PER_AREA = 3
RMS_WINDOW_MS = 20.0
START_Z = 4.0
END_Z = 2.0
MIN_DURATION_MS = 20.0
MAX_DURATION_MS = 200.0
ARTIFACT_Z = 20.0
EDGE_S = 1.0
RANDOM_SEED = 20260705


def robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    median = np.median(values)
    scale = 1.4826 * np.median(np.abs(values - median))
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        return np.zeros_like(values, dtype=float)
    return (values - median) / scale


def contiguous_regions(mask: np.ndarray):
    changes = np.diff(np.pad(np.asarray(mask, dtype=np.int8), (1, 1)))
    return zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))


def selected_channels(n_channels: int) -> np.ndarray:
    return np.unique(
        np.linspace(0, n_channels - 1, min(CHANNELS_PER_AREA, n_channels))
        .round()
        .astype(int)
    )


def npy_metadata(path: Path):
    with path.open("rb") as file:
        version = np.lib.format.read_magic(file)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(file)
        else:
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(file)
        return shape, fortran_order, dtype, file.tell()


def map_npy_channel(path: Path, channel: int, metadata):
    shape, fortran_order, dtype, data_offset = metadata
    if len(shape) != 2 or fortran_order:
        raise ValueError(f"Expected a C-order channel × time array: {path}")
    channel_offset = data_offset + channel * shape[1] * dtype.itemsize
    return np.memmap(
        path,
        dtype=dtype,
        mode="r",
        offset=channel_offset,
        shape=(shape[1],),
    )


def load_npy_channels(path: Path, channels: np.ndarray, metadata) -> np.ndarray:
    shape, fortran_order, dtype, data_offset = metadata
    if not fortran_order:
        traces = []
        for channel in channels:
            trace = map_npy_channel(path, int(channel), metadata)
            traces.append(np.asarray(trace).copy())
            trace._mmap.close()
        return np.stack(traces)

    # Fortran-order arrays store all channels for each time point together.
    # Stream once through the file and retain only the requested channels.
    n_channels, n_samples = shape
    output = np.empty((len(channels), n_samples), dtype=dtype)
    chunk_samples = 100_000
    with path.open("rb") as file:
        file.seek(data_offset)
        position = 0
        while position < n_samples:
            count = min(chunk_samples, n_samples - position)
            values = np.fromfile(file, dtype=dtype, count=n_channels * count)
            complete_samples = len(values) // n_channels
            if complete_samples:
                complete_values = values[: complete_samples * n_channels]
                block = complete_values.reshape(
                    (n_channels, complete_samples),
                    order="F",
                )
                output[:, position : position + complete_samples] = block[channels]
                position += complete_samples

            remainder = len(values) % n_channels
            if remainder:
                # A few converted files are truncated by less than one sample.
                # Preserve any available channel values and carry the previous
                # sample forward for missing values; this affects <2 ms.
                output[:, position] = output[:, position - 1]
                partial = values[-remainder:]
                for output_index, channel in enumerate(channels):
                    if channel < remainder:
                        output[output_index, position] = partial[channel]
                position += 1

            if len(values) < n_channels * count:
                # The declared NPY shape can exceed the physical file by a
                # handful of values. Fill the tiny missing tail continuously
                # so filtering remains finite.
                output[:, position:] = output[:, position - 1, None]
                break
    return output


def detect_rms_events(trace: np.ndarray) -> pd.DataFrame:
    x = np.asarray(trace, dtype=float)
    sos = signal.butter(4, RIPPLE_BAND, btype="bandpass", fs=FS, output="sos")
    filtered = signal.sosfiltfilt(sos, x)
    window = max(1, int(round(RMS_WINDOW_MS / 1000 * FS)))
    rms = np.sqrt(ndimage.uniform_filter1d(filtered**2, size=window))
    score = robust_z(rms)
    raw_score = np.abs(robust_z(x))

    min_samples = int(math.ceil(MIN_DURATION_MS / 1000 * FS))
    max_samples = int(math.floor(MAX_DURATION_MS / 1000 * FS))
    edge = int(round(EDGE_S * FS))
    rows = []
    for start, stop in contiguous_regions(score >= END_Z):
        duration = stop - start
        if start < edge or stop > len(x) - edge:
            continue
        if not min_samples <= duration <= max_samples:
            continue
        if np.max(score[start:stop]) < START_Z:
            continue
        if np.max(raw_score[start:stop]) > ARTIFACT_Z:
            continue
        peak = start + int(np.argmax(score[start:stop]))
        rows.append(
            {
                "start_s": start / FS,
                "stop_s": stop / FS,
                "center_s": peak / FS,
                "duration_ms": duration / FS * 1000,
                "peak_robust_z": float(score[peak]),
            }
        )
    return pd.DataFrame(rows)


def analyze_mouse(mouse: int, data_dir: Path) -> pd.DataFrame:
    if mouse == 12:
        raise ValueError("Mouse 12 is excluded because it duplicates mouse 11.")
    rows = []
    for area in AREAS:
        path = data_dir / str(mouse) / f"lfp_{area}.npy"
        metadata = npy_metadata(path)
        shape = metadata[0]
        duration_min = shape[1] / FS / 60
        channels = selected_channels(shape[0])
        traces = load_npy_channels(path, channels, metadata)
        for channel, trace in zip(channels, traces):
            events = detect_rms_events(trace)
            rows.append(
                {
                    "mouse": mouse,
                    "brain_area": area,
                    "channel": int(channel),
                    "n_channels_available": shape[0],
                    "duration_min": duration_min,
                    "n_events": len(events),
                    "events_per_min": len(events) / duration_min,
                    "median_duration_ms": (
                        float(events["duration_ms"].median())
                        if len(events)
                        else np.nan
                    ),
                    "median_peak_robust_z": (
                        float(events["peak_robust_z"].median())
                        if len(events)
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 20_000):
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.integers(0, len(x), size=(n_boot, len(x)))
    bootstrap = x[indices].mean(axis=1)
    low, high = np.percentile(bootstrap, [2.5, 97.5])
    return float(np.mean(x)), float(low), float(high)


def holm_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p) - rank) * p[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def summarize(mouse_results: pd.DataFrame):
    wide = mouse_results.pivot(
        index="mouse", columns="brain_area", values="events_per_min"
    )
    summary_rows = []
    for area in AREAS:
        mean, low, high = bootstrap_mean_ci(wide[area].to_numpy())
        summary_rows.append(
            {
                "brain_area": area,
                "mean_events_per_min": mean,
                "ci95_low": low,
                "ci95_high": high,
                "median_events_per_min": float(wide[area].median()),
                "n_mice": len(wide),
            }
        )

    friedman = stats.friedmanchisquare(*(wide[area] for area in AREAS))
    pairs = []
    raw_p = []
    for left, right in ((1, 2), (1, 3), (2, 3)):
        result = stats.wilcoxon(wide[left], wide[right], alternative="two-sided")
        raw_p.append(float(result.pvalue))
        pairs.append(
            {
                "left_area": left,
                "right_area": right,
                "mean_paired_difference": float((wide[left] - wide[right]).mean()),
                "p_raw": float(result.pvalue),
                "left_wins": int((wide[left] > wide[right]).sum()),
                "right_wins": int((wide[right] > wide[left]).sum()),
            }
        )
    for row, adjusted in zip(pairs, holm_adjust(raw_p)):
        row["p_holm"] = adjusted

    tests = {
        "friedman_statistic": float(friedman.statistic),
        "friedman_p": float(friedman.pvalue),
        "pairwise": pairs,
    }
    return pd.DataFrame(summary_rows), tests, wide


def plot_results(
    wide: pd.DataFrame, summary: pd.DataFrame, output_path: Path
):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(1, 4)
    for _, row in wide.iterrows():
        ax.plot(x, row.to_numpy(), color="0.75", linewidth=0.8, alpha=0.75)
        ax.scatter(x, row.to_numpy(), color="0.35", s=16, alpha=0.8)
    ordered = summary.set_index("brain_area").loc[list(AREAS)]
    means = ordered["mean_events_per_min"].to_numpy()
    yerr = np.vstack(
        [
            means - ordered["ci95_low"].to_numpy(),
            ordered["ci95_high"].to_numpy() - means,
        ]
    )
    ax.errorbar(
        x,
        means,
        yerr=yerr,
        color="#C43C39",
        marker="D",
        markersize=7,
        linewidth=2,
        capsize=5,
        zorder=5,
        label="Mean ± 95% bootstrap CI",
    )
    ax.set_xticks(x, [f"Area {area}" for area in AREAS])
    ax.set_ylabel("Ripple candidates/min")
    ax.set_title("Robust RMS ripple density across 17 unique mice")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/q1_rms"))
    parser.add_argument("--mice", type=int, nargs="+", default=list(VALID_MICE))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "mice"
    cache_dir.mkdir(exist_ok=True)

    frames = []
    mice = tuple(args.mice)
    if any(mouse not in VALID_MICE for mouse in mice):
        raise ValueError(f"Allowed mice are {VALID_MICE}; mouse 12 is excluded.")
    for mouse in mice:
        cache = cache_dir / f"mouse_{mouse}.csv"
        if cache.exists() and not args.force:
            print(f"Mouse {mouse}: cached", flush=True)
            frame = pd.read_csv(cache)
        else:
            print(f"Mouse {mouse}: analyzing", flush=True)
            frame = analyze_mouse(mouse, args.data_dir)
            frame.to_csv(cache, index=False)
        frames.append(frame)

    channel_results = pd.concat(frames, ignore_index=True)
    channel_results.to_csv(args.output_dir / "channel_results.csv", index=False)
    mouse_results = (
        channel_results.groupby(["mouse", "brain_area"], as_index=False)
        .agg(
            events_per_min=("events_per_min", "median"),
            min_channel_rate=("events_per_min", "min"),
            max_channel_rate=("events_per_min", "max"),
            median_event_duration_ms=("median_duration_ms", "median"),
            median_peak_robust_z=("median_peak_robust_z", "median"),
        )
    )
    mouse_results.to_csv(args.output_dir / "mouse_results.csv", index=False)
    summary, tests, wide = summarize(mouse_results)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "tests.json").write_text(
        json.dumps(tests, indent=2), encoding="utf-8"
    )
    plot_results(wide, summary, args.output_dir / "paired_mouse_rates.png")

    print("\nSummary", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(tests, indent=2), flush=True)


if __name__ == "__main__":
    main()
