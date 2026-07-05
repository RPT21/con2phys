"""Compare PridaLab rippl-AI CNN1D with the robust RMS detector.

The pretrained model expects 8 channels at 1250 Hz. Target LFP is therefore
polyphase-resampled from 500 to 1250 Hz, z-scored per channel, and passed to
the repository's best CNN1D SavedModel. TensorFlow's SavedModel API is used
because the repository predates Keras 3 and its direct load_model call is no
longer compatible.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import ndimage, signal
import tensorflow as tf

from run_q1_rms import (
    AREAS,
    FS,
    detect_rms_events,
    load_npy_channels,
    npy_metadata,
    robust_z,
)


RIPPL_FS = 1250
MODEL_THRESHOLD = 0.5
MODEL_SPECS = {
    1: {
        "path": Path(
            "tools/rippl-AI/optimized_models/CNN1D_1_Ch8_W60_Ts16_OGmodel12"
        ),
        "channels": 8,
        "timesteps": 16,
    },
    6: {
        "path": Path(
            "tools/rippl-AI/optimized_models/"
            "CNN1D_6_Ch1_W60_Ts40_Fp1.50_E50_TB32"
        ),
        "channels": 1,
        "timesteps": 40,
    },
}
MIN_DURATION_MS = 20.0
MAX_DURATION_MS = 200.0


def load_time_slice(path: Path, start: int, stop: int, metadata) -> np.ndarray:
    """Read all channels for a short time slice without mapping the full file."""

    shape, fortran_order, dtype, data_offset = metadata
    n_channels, n_samples = shape
    start = max(0, start)
    stop = min(n_samples, stop)
    count = stop - start
    if count <= 0:
        return np.empty((n_channels, 0), dtype=dtype)

    if fortran_order:
        with path.open("rb") as file:
            file.seek(data_offset + start * n_channels * dtype.itemsize)
            values = np.fromfile(file, dtype=dtype, count=n_channels * count)
        complete = len(values) // n_channels
        result = values[: complete * n_channels].reshape(
            (n_channels, complete),
            order="F",
        )
        if complete < count:
            padded = np.empty((n_channels, count), dtype=dtype)
            padded[:, :complete] = result
            padded[:, complete:] = result[:, -1, None]
            result = padded
        return result

    traces = []
    for channel in range(n_channels):
        channel_offset = (
            data_offset + channel * n_samples * dtype.itemsize
            + start * dtype.itemsize
        )
        trace = np.memmap(
            path,
            dtype=dtype,
            mode="r",
            offset=channel_offset,
            shape=(count,),
        )
        traces.append(np.asarray(trace).copy())
        trace._mmap.close()
    return np.stack(traces)


def select_pyramidal_channel(path: Path, metadata) -> tuple[int, np.ndarray]:
    """Approximate rippl-AI's pyramidal-layer selection using ripple power."""

    n_channels, n_samples = metadata[0]
    chunk_samples = min(int(60 * FS), n_samples)
    starts = np.linspace(
        0,
        max(0, n_samples - chunk_samples),
        5,
    ).round().astype(int)
    ripple_power_chunks = []
    for start in starts:
        block = load_time_slice(path, int(start), int(start + chunk_samples), metadata)
        frequencies, psd = signal.welch(
            block,
            fs=FS,
            nperseg=int(2 * FS),
            noverlap=int(FS),
            axis=-1,
            detrend="constant",
        )
        ripple_mask = (frequencies >= 100) & (frequencies <= 200)
        broad_mask = (frequencies >= 1) & (frequencies <= 240)
        ripple_power = np.trapezoid(
            psd[:, ripple_mask],
            frequencies[ripple_mask],
            axis=1,
        )
        broad_power = np.trapezoid(
            psd[:, broad_mask],
            frequencies[broad_mask],
            axis=1,
        )
        ripple_power_chunks.append(ripple_power)
    # Match rippl-AI's documented Neuropixels recommendation: select the
    # putative pyramidal contact using maximal net ripple-band power.
    score = np.median(np.stack(ripple_power_chunks), axis=0)
    score = ndimage.median_filter(score, size=3, mode="nearest")
    return int(np.nanargmax(score)), score


def model_channel_indices(pyramidal_channel: int, n_channels: int) -> np.ndarray:
    """Choose eight spatially ordered contacts around the power maximum."""

    # The repository recommends pyr + [-8,-6,-4,-2,0,2,4,6].
    # Shift the 15-contact window at array boundaries.
    start = int(np.clip(pyramidal_channel - 8, 0, max(0, n_channels - 15)))
    indices = start + np.arange(0, 15, 2)
    indices = np.clip(indices, 0, n_channels - 1)
    # Target arrays are documented deepest-to-superficial. rippl-AI expects
    # upper-to-lower spatial order, so present them in reverse index order.
    return indices[::-1].astype(int)


def load_saved_model(model_path: Path):
    model = tf.saved_model.load(str(model_path))
    return model.signatures["serving_default"]


def preprocess_for_rippl_ai(traces: np.ndarray) -> np.ndarray:
    """Match repository preprocessing, adding the requested upsampling."""

    data = np.asarray(traces.T, dtype=np.float32)
    data = signal.resample_poly(data, up=5, down=2, axis=0)
    mean = np.mean(data, axis=0, dtype=np.float64)
    std = np.std(data, axis=0, dtype=np.float64)
    std[~np.isfinite(std) | (std <= np.finfo(float).eps)] = 1.0
    data = ((data - mean) / std).astype(np.float32)
    return data


def predict_probability(
    normalized_lfp: np.ndarray,
    inference_function,
    model_channels: int,
    model_timesteps: int,
    batch_windows: int = 4096,
) -> np.ndarray:
    usable = len(normalized_lfp) - len(normalized_lfp) % model_timesteps
    windows = normalized_lfp[:usable].reshape(
        -1,
        model_timesteps,
        model_channels,
    )
    probability = np.zeros(usable, dtype=np.float32)
    for start in range(0, len(windows), batch_windows):
        stop = min(len(windows), start + batch_windows)
        prediction = inference_function(
            conv1d_input=tf.convert_to_tensor(windows[start:stop])
        )["dense"].numpy()
        probability[
            start * model_timesteps : stop * model_timesteps
        ] = np.repeat(prediction.reshape(-1), model_timesteps)
    if usable < len(normalized_lfp):
        probability = np.pad(
            probability,
            (0, len(normalized_lfp) - usable),
            mode="constant",
        )
    return probability


def threshold_intervals(
    probability: np.ndarray,
    threshold: float = MODEL_THRESHOLD,
    duration_filter: bool = True,
) -> pd.DataFrame:
    above = probability >= threshold
    changes = np.diff(np.pad(above.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    rows = []
    for start, stop in zip(starts, stops):
        duration_ms = (stop - start) / RIPPL_FS * 1000
        if duration_filter and not MIN_DURATION_MS <= duration_ms <= MAX_DURATION_MS:
            continue
        peak = start + int(np.argmax(probability[start:stop]))
        rows.append(
            {
                "start_s": start / RIPPL_FS,
                "stop_s": stop / RIPPL_FS,
                "center_s": peak / RIPPL_FS,
                "duration_ms": duration_ms,
                "peak_probability": float(probability[peak]),
            }
        )
    return pd.DataFrame(rows)


def matched_fraction(reference_times, candidate_times, tolerance_s=0.100):
    reference = np.sort(np.asarray(reference_times, dtype=float))
    candidate = np.sort(np.asarray(candidate_times, dtype=float))
    if len(reference) == 0:
        return np.nan
    if len(candidate) == 0:
        return 0.0
    index = np.searchsorted(candidate, reference)
    distance = np.full(len(reference), np.inf)
    right = index < len(candidate)
    distance[right] = np.abs(reference[right] - candidate[index[right]])
    left = index > 0
    distance[left] = np.minimum(
        distance[left],
        np.abs(reference[left] - candidate[index[left] - 1]),
    )
    return float(np.mean(distance <= tolerance_s))


def analyze_mouse(
    mouse: int,
    data_dir: Path,
    output_dir: Path,
    model_number: int,
):
    if mouse == 12:
        raise ValueError("Mouse 12 duplicates Mouse 11 and is excluded.")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_spec = MODEL_SPECS[model_number]
    inference = load_saved_model(model_spec["path"])
    rms_area = pd.read_csv("results/q1_rms/mouse_results.csv")
    rms_area = rms_area[rms_area["mouse"] == mouse].set_index("brain_area")

    summary_rows = []
    sensitivity_rows = []
    event_tables = {}
    representative_traces = {}
    start_time = time.perf_counter()
    for area in AREAS:
        path = data_dir / str(mouse) / f"lfp_{area}.npy"
        metadata = npy_metadata(path)
        duration_min = metadata[0][1] / FS / 60
        pyramidal, power_score = select_pyramidal_channel(path, metadata)
        if model_spec["channels"] == 1:
            channels = np.array([pyramidal], dtype=int)
        else:
            channels = model_channel_indices(pyramidal, metadata[0][0])
        print(
            f"Area {area}: power maximum channel {pyramidal}; "
            f"CNN channels {channels.tolist()}",
            flush=True,
        )
        traces = load_npy_channels(path, channels, metadata)
        normalized = preprocess_for_rippl_ai(traces)
        probability = predict_probability(
            normalized,
            inference,
            model_channels=model_spec["channels"],
            model_timesteps=model_spec["timesteps"],
        )
        native_events = threshold_intervals(probability, duration_filter=False)
        cnn_events = threshold_intervals(probability, duration_filter=True)
        for threshold in (0.1, 0.2, 0.3, 0.4, 0.5):
            threshold_events = threshold_intervals(
                probability,
                threshold=threshold,
                duration_filter=True,
            )
            sensitivity_rows.append(
                {
                    "mouse": mouse,
                    "brain_area": area,
                    "threshold": threshold,
                    "n_events": len(threshold_events),
                    "events_per_min": len(threshold_events) / duration_min,
                }
            )

        # Compare temporal overlap on the same central/pyramidal contact.
        pyramidal_trace = load_npy_channels(
            path,
            np.array([pyramidal]),
            metadata,
        )[0]
        rms_events = detect_rms_events(pyramidal_trace)
        representative_traces[area] = pyramidal_trace
        event_tables[area] = {"cnn": cnn_events, "rms": rms_events}

        cnn_events.to_csv(
            output_dir / f"mouse_{mouse}_area_{area}_cnn1d_events.csv",
            index=False,
        )
        native_events.to_csv(
            output_dir / f"mouse_{mouse}_area_{area}_cnn1d_native_events.csv",
            index=False,
        )
        pd.DataFrame(
            {"probability": probability[::25]}
        ).to_csv(
            output_dir / f"mouse_{mouse}_area_{area}_probability_50hz.csv.gz",
            index=False,
            compression="gzip",
        )
        summary_rows.append(
            {
                "mouse": mouse,
                "model_number": model_number,
                "brain_area": area,
                "pyramidal_channel": pyramidal,
                "cnn_channels": " ".join(map(str, channels)),
                "duration_min": duration_min,
                "cnn_native_events": len(native_events),
                "cnn_duration_filtered_events": len(cnn_events),
                "cnn_events_per_min": len(cnn_events) / duration_min,
                "cnn_median_duration_ms": (
                    float(cnn_events["duration_ms"].median())
                    if len(cnn_events)
                    else np.nan
                ),
                "same_channel_rms_events_per_min": len(rms_events) / duration_min,
                "final_rms_area_events_per_min": float(
                    rms_area.loc[area, "events_per_min"]
                ),
                "fraction_rms_matched_by_cnn": matched_fraction(
                    rms_events.get("center_s", pd.Series(dtype=float)),
                    cnn_events.get("center_s", pd.Series(dtype=float)),
                ),
                "fraction_cnn_matched_by_rms": matched_fraction(
                    cnn_events.get("center_s", pd.Series(dtype=float)),
                    rms_events.get("center_s", pd.Series(dtype=float)),
                ),
                "probability_p99": float(np.quantile(probability, 0.99)),
                "probability_p999": float(np.quantile(probability, 0.999)),
                "probability_max": float(np.max(probability)),
            }
        )
        del traces, normalized, probability

    results = pd.DataFrame(summary_rows)
    results["elapsed_seconds"] = time.perf_counter() - start_time
    results.to_csv(output_dir / f"mouse_{mouse}_comparison.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(
        output_dir / f"mouse_{mouse}_threshold_sensitivity.csv",
        index=False,
    )
    plot_path = plot_comparison(
        mouse,
        representative_traces,
        event_tables,
        output_dir,
        model_number,
    )
    return results, plot_path


def plot_comparison(
    mouse: int,
    traces: dict,
    event_tables: dict,
    output_dir: Path,
    model_number: int,
    window_s: float = 3.0,
) -> Path:
    all_events = []
    for area, methods in event_tables.items():
        for method, frame in methods.items():
            for center in frame.get("center_s", pd.Series(dtype=float)):
                all_events.append((float(center), area, method))
    if not all_events:
        raise RuntimeError("No events available for the comparison plot.")

    starts = [max(0, event[0] - window_s / 2) for event in all_events]
    best_start, best_score = starts[0], (-1, -1)
    for start in starts:
        inside = [
            item for item in all_events if start <= item[0] <= start + window_s
        ]
        score = (len({(area, method) for _, area, method in inside}), len(inside))
        if score > best_score:
            best_start, best_score = start, score
    best_stop = best_start + window_s

    colors = {"rms": "#0072B2", "cnn": "#D55E00"}
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    sos = signal.butter(4, (100, 200), btype="bandpass", fs=FS, output="sos")
    for area, ax in zip(AREAS, axes):
        trace = traces[area]
        pad = int(FS)
        start = max(0, int(best_start * FS) - pad)
        stop = min(len(trace), int(best_stop * FS) + pad)
        filtered = signal.sosfiltfilt(sos, trace[start:stop])
        display = robust_z(filtered)
        time_axis = np.arange(start, stop) / FS
        mask = (time_axis >= best_start) & (time_axis <= best_stop)
        ax.plot(time_axis[mask], display[mask], color="0.25", linewidth=0.7)
        low, high = np.percentile(display[mask], [1, 99])
        separation = max(0.8, 0.12 * (high - low))
        rows = {"rms": high + 1.2 * separation, "cnn": high + 2.3 * separation}
        for method in ("rms", "cnn"):
            for event in event_tables[area][method].itertuples(index=False):
                if event.stop_s < best_start or event.start_s > best_stop:
                    continue
                ax.hlines(
                    rows[method],
                    max(best_start, event.start_s),
                    min(best_stop, event.stop_s),
                    color=colors[method],
                    linewidth=5,
                )
        ax.set_ylim(low - separation, high + 3 * separation)
        ax.set_ylabel("Robust z")
        ax.set_title(f"Area {area}")
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Session time (s)")
    axes[0].legend(
        handles=[
            Line2D([0], [0], color=colors["rms"], linewidth=5, label="Robust RMS"),
            Line2D(
                [0],
                [0],
                color=colors["cnn"],
                linewidth=5,
                label=f"rippl-AI CNN1D model {model_number}",
            ),
            Line2D([0], [0], color="0.25", linewidth=1, label="100–200 Hz LFP"),
        ],
        frameon=False,
        ncol=3,
        loc="upper right",
    )
    fig.suptitle(
        f"Mouse {mouse}: RMS versus rippl-AI CNN1D model {model_number}",
        y=0.995,
    )
    fig.tight_layout()
    path = output_dir / f"mouse_{mouse}_rms_vs_rippl_ai.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mouse", type=int, default=1)
    parser.add_argument("--model-number", type=int, choices=(1, 6), default=1)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/q1_rippl_ai"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results, plot_path = analyze_mouse(
        args.mouse,
        args.data_dir,
        args.output_dir,
        args.model_number,
    )
    print(results.to_string(index=False))
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
