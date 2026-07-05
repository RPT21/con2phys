"""Compare different ripple detection pipelines for Question 1."""

import math
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import ndimage, signal, stats
import concurrent.futures
import time

FS = 500.0
VALID_MICE = tuple(range(1, 12)) + tuple(range(13, 19))
AREAS = (1, 2, 3)
CHANNELS_PER_AREA = 3
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
        raise ValueError(f"Expected a C-order channel x time array: {path}")
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
    # Fortran-order
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
                output[:, position] = output[:, position - 1]
                partial = values[-remainder:]
                for output_index, channel in enumerate(channels):
                    if channel < remainder:
                        output[output_index, position] = partial[channel]
                position += 1
            if len(values) < n_channels * count:
                output[:, position:] = output[:, position - 1, None]
                break
    return output

def run_detection(trace: np.ndarray, pipeline: str):
    """Detect events and calculate ripple band power for a single trace."""
    x = np.asarray(trace, dtype=float)
    
    # Define parameters based on pipeline
    if pipeline == "classic_band":
        # 150-240 Hz Bandpass, RMS envelope (20ms window)
        ripple_band = (150.0, 240.0)
        sos = signal.butter(4, ripple_band, btype="bandpass", fs=FS, output="sos")
        filtered = signal.sosfiltfilt(sos, x)
        window = max(1, int(round(20.0 / 1000 * FS)))
        envelope = np.sqrt(ndimage.uniform_filter1d(filtered**2, size=window))
        score = robust_z(envelope)
        power = np.mean(filtered**2)
    elif pipeline == "rectified":
        # 100-200 Hz Bandpass, absolute value envelope with 10 ms moving average
        ripple_band = (100.0, 200.0)
        sos = signal.butter(4, ripple_band, btype="bandpass", fs=FS, output="sos")
        filtered = signal.sosfiltfilt(sos, x)
        envelope = np.abs(filtered)
        window = max(1, int(round(10.0 / 1000 * FS)))
        smoothed = ndimage.uniform_filter1d(envelope, size=window)
        score = robust_z(smoothed)
        power = np.mean(filtered**2)
    else:
        # Default RMS: 100-200 Hz Bandpass, RMS envelope (20ms window)
        ripple_band = (100.0, 200.0)
        sos = signal.butter(4, ripple_band, btype="bandpass", fs=FS, output="sos")
        filtered = signal.sosfiltfilt(sos, x)
        window = max(1, int(round(20.0 / 1000 * FS)))
        envelope = np.sqrt(ndimage.uniform_filter1d(filtered**2, size=window))
        score = robust_z(envelope)
        power = np.mean(filtered**2)

    raw_score = np.abs(robust_z(x))
    min_samples = int(math.ceil(20.0 / 1000 * FS))
    max_samples = int(math.floor(200.0 / 1000 * FS))
    edge = int(round(1.0 * FS))
    
    n_events = 0
    for start, stop in contiguous_regions(score >= 2.0):
        duration = stop - start
        if start < edge or stop > len(x) - edge:
            continue
        if not min_samples <= duration <= max_samples:
            continue
        if np.max(score[start:stop]) < 4.0:
            continue
        if np.max(raw_score[start:stop]) > 20.0:
            continue
        n_events += 1
        
    return n_events, power

def analyze_mouse_pipeline(mouse: int, data_dir: Path, pipeline: str) -> list[dict]:
    if mouse == 12:
        return []
    rows = []
    for area in AREAS:
        path = data_dir / str(mouse) / f"lfp_{area}.npy"
        metadata = npy_metadata(path)
        shape = metadata[0]
        duration_min = shape[1] / FS / 60
        channels = selected_channels(shape[0])
        traces = load_npy_channels(path, channels, metadata)
        
        for idx, (channel, trace) in enumerate(zip(channels, traces)):
            n_events, power = run_detection(trace, pipeline)
            rows.append({
                "mouse": mouse,
                "brain_area": area,
                "channel": int(channel),
                "channel_idx": idx, # index within selected channels
                "duration_min": duration_min,
                "n_events": n_events,
                "events_per_min": n_events / duration_min,
                "ripple_power": power
            })
    return rows

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

def run_pipeline_analysis(channel_results: pd.DataFrame, pipeline_name: str, aggregation_mode: str):
    """Aggregate channel results and compute group statistics."""
    # Aggregation
    if aggregation_mode == "median":
        mouse_results = (
            channel_results.groupby(["mouse", "brain_area"], as_index=False)
            .agg(events_per_min=("events_per_min", "median"))
        )
    elif aggregation_mode == "max_rate":
        mouse_results = (
            channel_results.groupby(["mouse", "brain_area"], as_index=False)
            .agg(events_per_min=("events_per_min", "max"))
        )
    elif aggregation_mode == "max_power":
        # Group by mouse and area, and select the channel with highest ripple_power
        idx = channel_results.groupby(["mouse", "brain_area"])["ripple_power"].idxmax()
        mouse_results = channel_results.loc[idx].reset_index(drop=True)
    else:
        raise ValueError(f"Unknown aggregation mode: {aggregation_mode}")

    # Pivot to wide format
    wide = mouse_results.pivot(
        index="mouse", columns="brain_area", values="events_per_min"
    )
    
    # Group summaries and bootstrap CIs
    summary_rows = []
    for area in AREAS:
        mean, low, high = bootstrap_mean_ci(wide[area].to_numpy())
        summary_rows.append({
            "brain_area": area,
            "mean_events_per_min": mean,
            "ci95_low": low,
            "ci95_high": high,
            "median_events_per_min": float(wide[area].median()),
            "n_mice": len(wide)
        })
    summary_df = pd.DataFrame(summary_rows)

    # Statistical tests
    friedman = stats.friedmanchisquare(*(wide[area] for area in AREAS))
    pairs = []
    raw_p = []
    for left, right in ((1, 2), (1, 3), (2, 3)):
        result = stats.wilcoxon(wide[left], wide[right], alternative="two-sided")
        raw_p.append(float(result.pvalue))
        pairs.append({
            "left_area": left,
            "right_area": right,
            "mean_paired_difference": float((wide[left] - wide[right]).mean()),
            "p_raw": float(result.pvalue),
            "left_wins": int((wide[left] > wide[right]).sum()),
            "right_wins": int((wide[right] > wide[left]).sum()),
        })
    for row, adjusted in zip(pairs, holm_adjust(raw_p)):
        row["p_holm"] = adjusted

    tests = {
        "friedman_statistic": float(friedman.statistic),
        "friedman_p": float(friedman.pvalue),
        "pairwise": pairs,
    }
    
    return summary_df, tests, wide

def process_pipeline(pipeline_name: str, detection_pipeline: str, aggregation_mode: str, data_dir: Path):
    print(f"\n--- Running Pipeline: {pipeline_name} ---")
    start_time = time.time()
    
    # Run detection on all mice in parallel
    results_list = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(analyze_mouse_pipeline, mouse, data_dir, detection_pipeline): mouse
            for mouse in VALID_MICE
        }
        for future in concurrent.futures.as_completed(futures):
            mouse = futures[future]
            try:
                rows = future.result()
                results_list.extend(rows)
                print(f"Mouse {mouse} done", flush=True)
            except Exception as exc:
                print(f"Mouse {mouse} generated an exception: {exc}", flush=True)
                
    df = pd.DataFrame(results_list)
    summary, tests, wide = run_pipeline_analysis(df, pipeline_name, aggregation_mode)
    
    print(f"Completed in {time.time() - start_time:.2f} seconds.")
    print("Summary:")
    print(summary.to_string(index=False))
    print("Friedman p-value:", tests["friedman_p"])
    for pair in tests["pairwise"]:
        print(f"Area {pair['left_area']} vs Area {pair['right_area']}: p_raw = {pair['p_raw']:.4f}, p_holm = {pair['p_holm']:.4f} (wins: {pair['left_wins']} vs {pair['right_wins']})")
        
    return summary, tests

def main():
    data_dir = Path("data")
    results_dir = Path("results/q1_pipeline_comparison")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    pipelines = [
        {
            "name": "Pipeline 1: Original RMS (100-200Hz, RMS, Median Channel)",
            "det": "default",
            "agg": "median"
        },
        {
            "name": "Pipeline 2: Classic Band (150-240Hz, RMS, Median Channel)",
            "det": "classic_band",
            "agg": "median"
        },
        {
            "name": "Pipeline 3: Rectified Envelope (100-200Hz, Rectified, Median Channel)",
            "det": "rectified",
            "agg": "median"
        },
        {
            "name": "Pipeline 4: Max Rate Channel (100-200Hz, RMS, Max Channel)",
            "det": "default",
            "agg": "max_rate"
        },
        {
            "name": "Pipeline 5: Max Power Channel (100-200Hz, RMS, Max Power Channel)",
            "det": "default",
            "agg": "max_power"
        }
    ]
    
    all_results = {}
    for p in pipelines:
        summary, tests = process_pipeline(p["name"], p["det"], p["agg"], data_dir)
        all_results[p["name"]] = {
            "summary": summary.to_dict(orient="records"),
            "tests": tests
        }
        
    # Save results to a json file
    with open(results_dir / "comparison_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll pipeline results saved to {results_dir / 'comparison_results.json'}")

if __name__ == "__main__":
    main()
