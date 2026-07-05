"""CPU-friendly supervised ripple-candidate classifier for Question 1.

The model is trained on a small, session-grouped subset of the public
Campbell & Murphy (2025) mouse Neuropixels SWR event tables.  The 1.3 GB OSF
ZIP is accessed with HTTP byte ranges, so only selected compressed CSV files
are downloaded (normally tens of MB).

This is an event-quality classifier, not an end-to-end waveform detector.
Broad candidates are generated from 100-200 Hz activity and the classifier
then scores physiology-inspired, dimensionless event features.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import re
import time
import urllib.request
import zipfile
from collections import OrderedDict, defaultdict
from pathlib import Path

# Keep training polite on a CPU-only laptop. These need to be set before
# NumPy/scikit-learn initialize their numerical backends.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import joblib
import numpy as np
import pandas as pd
from scipy import ndimage, signal
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from run_q1_rms import (
    FS,
    AREAS,
    detect_rms_events,
    load_npy_channels,
    npy_metadata,
    robust_z,
    selected_channels,
)


RANDOM_SEED = 20260705
SOURCE_ARCHIVE_URL = (
    "https://files.ca-1.osf.io/v1/resources/9gm6x/providers/"
    "osfstorage/68c89cf8f043ce674f4141eb"
)
SOURCE_ARCHIVE_SIZE = 1_308_294_944
SOURCE_DOI = "10.17605/OSF.IO/9GM6X"

# All features have a direct analogue that can be computed from the target
# LFP. Artifact flags and channel identity are intentionally excluded.
FEATURES = (
    "duration",
    "power_max_zscore",
    "power_median_zscore",
    "power_mean_zscore",
    "power_min_zscore",
    "power_90th_percentile",
    "sw_peak_power",
    "envelope_mean_zscore",
    "envelope_median_zscore",
    "envelope_max_zscore",
    "envelope_min_zscore",
    "envelope_area",
    "envelope_total_energy",
    "envelope_90th_percentile",
    "gamma_overlap_percent",
)

DATASET_PREFIXES = {
    "allen_visbehave_swr_murphylab2024": "allen_visual_behavior",
    "allen_viscoding_swr_murphylab2024": "allen_visual_coding",
    "ibl_swr_murphylab2024": "ibl_decision_making",
}


class HTTPRangeReader(io.RawIOBase):
    """Small seekable HTTP reader used by zipfile.ZipFile."""

    def __init__(
        self,
        url: str,
        size: int,
        block_size: int = 1 << 20,
        max_cached_blocks: int = 12,
    ):
        self.url = url
        self.size = size
        self.block_size = block_size
        self.max_cached_blocks = max_cached_blocks
        self.position = 0
        self.cache: OrderedDict[int, bytes] = OrderedDict()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            new_position = offset
        elif whence == io.SEEK_CUR:
            new_position = self.position + offset
        elif whence == io.SEEK_END:
            new_position = self.size + offset
        else:
            raise ValueError(f"Unknown whence: {whence}")
        if new_position < 0:
            raise ValueError("Negative seek position")
        self.position = new_position
        return self.position

    def _get_block(self, block_index: int) -> bytes:
        if block_index in self.cache:
            self.cache.move_to_end(block_index)
            return self.cache[block_index]

        start = block_index * self.block_size
        stop = min(self.size, start + self.block_size)
        request = urllib.request.Request(
            self.url,
            headers={"Range": f"bytes={start}-{stop - 1}"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read()
        if len(data) != stop - start:
            raise IOError(
                f"Expected {stop - start} bytes from OSF, received {len(data)}"
            )
        self.cache[block_index] = data
        if len(self.cache) > self.max_cached_blocks:
            self.cache.popitem(last=False)
        return data

    def read(self, size=-1):
        if size is None or size < 0:
            size = self.size - self.position
        if size == 0 or self.position >= self.size:
            return b""

        end = min(self.size, self.position + size)
        chunks = []
        while self.position < end:
            block_index = self.position // self.block_size
            block_start = block_index * self.block_size
            block_stop = min(self.size, block_start + self.block_size)
            block = self._get_block(block_index)
            count = min(end - self.position, block_stop - self.position)
            local_start = self.position - block_start
            chunks.append(block[local_start : local_start + count])
            self.position += count
        return b"".join(chunks)


def source_identity(archive_path: str) -> tuple[str, str]:
    dataset_prefix = archive_path.split("/", 1)[0]
    dataset = DATASET_PREFIXES[dataset_prefix]
    match = re.search(r"/swrs_session_([^/]+)/", archive_path)
    if match is None:
        raise ValueError(f"Cannot parse session from {archive_path}")
    return dataset, match.group(1)


def select_source_tables(
    names: list[str],
    sessions_per_dataset: int,
    tables_per_session: int,
) -> list[dict]:
    grouped: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for name in names:
        if not name.endswith("_putative_swr_events.csv.gz"):
            continue
        dataset, session_id = source_identity(name)
        grouped[dataset][session_id].append(name)

    rng = np.random.default_rng(RANDOM_SEED)
    selected = []
    for dataset in sorted(grouped):
        sessions = np.array(sorted(grouped[dataset]), dtype=object)
        rng.shuffle(sessions)
        for session_id in sessions[:sessions_per_dataset]:
            table_names = sorted(grouped[dataset][str(session_id)])
            for name in table_names[:tables_per_session]:
                selected.append(
                    {
                        "dataset": dataset,
                        "session_id": str(session_id),
                        "archive_path": name,
                    }
                )
    return selected


def cache_source_tables(
    cache_dir: Path,
    sessions_per_dataset: int = 6,
    tables_per_session: int = 2,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.csv"

    remote = HTTPRangeReader(SOURCE_ARCHIVE_URL, SOURCE_ARCHIVE_SIZE)
    with zipfile.ZipFile(remote) as archive:
        selection = select_source_tables(
            archive.namelist(),
            sessions_per_dataset=sessions_per_dataset,
            tables_per_session=tables_per_session,
        )
        for row in selection:
            safe_name = (
                f"{row['dataset']}__{row['session_id']}__"
                f"{Path(row['archive_path']).name}"
            )
            local_path = cache_dir / safe_name
            row["local_path"] = str(local_path)
            if not local_path.exists():
                print(f"Downloading selected table: {safe_name}", flush=True)
                local_path.write_bytes(archive.read(row["archive_path"]))
            row["compressed_bytes"] = local_path.stat().st_size

    manifest = pd.DataFrame(selection)
    manifest.to_csv(manifest_path, index=False)
    return manifest


def high_confidence_label(frame: pd.DataFrame) -> np.ndarray:
    """Reproduce the publication's recommended high-confidence criteria."""

    return (
        frame["power_max_zscore"].between(3.0, 10.0)
        & frame["sw_exceeds_threshold"].astype(bool)
        & ~frame["overlaps_with_gamma"].astype(bool)
        & ~frame["overlaps_with_movement"].astype(bool)
    ).to_numpy()


def load_training_events(
    manifest: pd.DataFrame,
    max_per_class_per_session: int = 1_000,
) -> pd.DataFrame:
    frames = []
    for row in manifest.itertuples(index=False):
        source = pd.read_csv(row.local_path, compression="gzip", index_col=0)
        missing = sorted(set(FEATURES) - set(source.columns))
        if missing:
            raise ValueError(f"{row.local_path} is missing features: {missing}")
        frame = source.loc[:, list(FEATURES)].replace([np.inf, -np.inf], np.nan)
        frame["label"] = high_confidence_label(source)
        frame["dataset"] = row.dataset
        frame["session_id"] = str(row.session_id)
        frames.append(frame)

    events = pd.concat(frames, ignore_index=True)
    rng = np.random.default_rng(RANDOM_SEED)
    sampled = []
    for session_id, frame in events.groupby("session_id", sort=True):
        positives = frame[frame["label"]]
        negatives = frame[~frame["label"]]
        count = min(max_per_class_per_session, len(positives), len(negatives))
        if count < 20:
            continue
        positive_idx = rng.choice(positives.index, size=count, replace=False)
        negative_idx = rng.choice(negatives.index, size=count, replace=False)
        sampled.append(events.loc[np.r_[positive_idx, negative_idx]])
    if not sampled:
        raise RuntimeError("No source sessions contained enough examples of both classes.")
    return pd.concat(sampled, ignore_index=True)


def make_model() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler(quantile_range=(10, 90))),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    learning_rate=0.07,
                    max_iter=80,
                    max_leaf_nodes=15,
                    min_samples_leaf=30,
                    l2_regularization=1.0,
                    early_stopping=True,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def choose_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    # precision_recall_curve has one extra precision/recall point.
    return float(thresholds[int(np.nanargmax(f1[:-1]))])


def metric_row(labels, probabilities, threshold, fold):
    predictions = probabilities >= threshold
    return {
        "fold": fold,
        "n_events": len(labels),
        "positive_fraction": float(np.mean(labels)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }


def train_source_model(
    output_dir: Path,
    cache_dir: Path,
    sessions_per_dataset: int = 6,
    tables_per_session: int = 2,
    max_per_class_per_session: int = 1_000,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = cache_source_tables(
        cache_dir,
        sessions_per_dataset=sessions_per_dataset,
        tables_per_session=tables_per_session,
    )
    events = load_training_events(
        manifest,
        max_per_class_per_session=max_per_class_per_session,
    )
    events.to_csv(output_dir / "source_training_sample.csv.gz", index=False)

    x = events.loc[:, FEATURES]
    y = events["label"].to_numpy(dtype=bool)
    groups = events["session_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    out_of_fold = np.full(len(events), np.nan)
    fold_assignment = np.full(len(events), -1)
    fold_rows = []

    for fold, (train_index, test_index) in enumerate(
        splitter.split(x, y, groups),
        start=1,
    ):
        model = make_model()
        model.fit(x.iloc[train_index], y[train_index])
        probability = model.predict_proba(x.iloc[test_index])[:, 1]
        out_of_fold[test_index] = probability
        fold_assignment[test_index] = fold
        fold_threshold = choose_threshold(y[test_index], probability)
        fold_rows.append(
            metric_row(
                y[test_index],
                probability,
                fold_threshold,
                fold,
            )
        )
        print(
            f"Fold {fold}: AP={fold_rows[-1]['average_precision']:.3f}, "
            f"F1={fold_rows[-1]['f1']:.3f}",
            flush=True,
        )

    threshold = choose_threshold(y, out_of_fold)
    overall = metric_row(y, out_of_fold, threshold, "out_of_fold")
    metrics = pd.DataFrame(fold_rows + [overall])
    metrics.to_csv(output_dir / "source_grouped_cv_metrics.csv", index=False)

    oof = events.loc[:, ["dataset", "session_id", "label"]].copy()
    oof["fold"] = fold_assignment
    oof["probability"] = out_of_fold
    oof.to_csv(output_dir / "source_out_of_fold_predictions.csv.gz", index=False)

    final_model = make_model()
    final_model.fit(x, y)
    bundle = {
        "model": final_model,
        "features": FEATURES,
        "threshold": threshold,
        "source_doi": SOURCE_DOI,
        "training_events": len(events),
        "training_sessions": int(events["session_id"].nunique()),
        "source_datasets": sorted(events["dataset"].unique().tolist()),
        "random_seed": RANDOM_SEED,
    }
    joblib.dump(bundle, output_dir / "source_event_classifier.joblib")
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as file:
        json.dump(
            {key: value for key, value in bundle.items() if key != "model"},
            file,
            indent=2,
        )
    print(
        f"Source model: {len(events)} events from "
        f"{events['session_id'].nunique()} sessions; threshold={threshold:.3f}",
        flush=True,
    )
    return bundle


def contiguous_regions(mask: np.ndarray):
    changes = np.diff(np.pad(np.asarray(mask, dtype=np.int8), (1, 1)))
    return zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))


def target_candidate_features(trace: np.ndarray) -> pd.DataFrame:
    """Generate broad candidates and source-compatible features at 500 Hz."""

    x = np.asarray(trace, dtype=float)
    ripple_sos = signal.butter(4, (100, 200), btype="bandpass", fs=FS, output="sos")
    sharp_wave_sos = signal.butter(4, (8, 40), btype="bandpass", fs=FS, output="sos")
    gamma_sos = signal.butter(4, (20, 80), btype="bandpass", fs=FS, output="sos")

    ripple = signal.sosfiltfilt(ripple_sos, x)
    envelope = np.abs(signal.hilbert(ripple))
    envelope = ndimage.uniform_filter1d(envelope, size=max(1, int(0.010 * FS)))
    envelope_z = robust_z(envelope)
    power_z = robust_z(envelope**2)

    sharp_wave = signal.sosfiltfilt(sharp_wave_sos, x)
    sharp_wave_envelope = np.abs(signal.hilbert(sharp_wave))
    sharp_wave_envelope = ndimage.uniform_filter1d(
        sharp_wave_envelope,
        size=max(1, int(0.020 * FS)),
    )
    sharp_wave_power_z = robust_z(sharp_wave_envelope**2)

    gamma = signal.sosfiltfilt(gamma_sos, x)
    gamma_envelope = np.abs(signal.hilbert(gamma))
    gamma_envelope = ndimage.uniform_filter1d(
        gamma_envelope,
        size=max(1, int(0.020 * FS)),
    )
    gamma_power_z = robust_z(gamma_envelope**2)
    raw_z = np.abs(robust_z(x))

    rows = []
    min_samples = int(np.ceil(0.010 * FS))
    max_samples = int(np.floor(0.250 * FS))
    edge = int(FS)
    for start, stop in contiguous_regions(envelope_z >= 1.0):
        duration_samples = stop - start
        if start < edge or stop > len(x) - edge:
            continue
        if not min_samples <= duration_samples <= max_samples:
            continue
        if np.max(envelope_z[start:stop]) < 2.0:
            continue
        if np.max(raw_z[start:stop]) > 30.0:
            continue

        event_power = power_z[start:stop]
        event_envelope = envelope_z[start:stop]
        peak = start + int(np.argmax(event_power))
        rows.append(
            {
                "start_s": start / FS,
                "stop_s": stop / FS,
                "center_s": peak / FS,
                "duration": duration_samples / FS,
                "power_max_zscore": float(np.max(event_power)),
                "power_median_zscore": float(np.median(event_power)),
                "power_mean_zscore": float(np.mean(event_power)),
                "power_min_zscore": float(np.min(event_power)),
                "power_90th_percentile": float(
                    np.percentile(event_power, 90)
                ),
                "sw_peak_power": float(
                    np.max(sharp_wave_power_z[start:stop])
                ),
                "envelope_mean_zscore": float(np.mean(event_envelope)),
                "envelope_median_zscore": float(np.median(event_envelope)),
                "envelope_max_zscore": float(np.max(event_envelope)),
                "envelope_min_zscore": float(np.min(event_envelope)),
                "envelope_area": float(
                    np.trapezoid(event_envelope, dx=1 / FS)
                ),
                "envelope_total_energy": float(
                    np.sum(event_envelope**2) / FS
                ),
                "envelope_90th_percentile": float(
                    np.percentile(event_envelope, 90)
                ),
                "gamma_overlap_percent": float(
                    100 * np.mean(gamma_power_z[start:stop] >= 3.0)
                ),
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
    indices = np.searchsorted(candidate, reference)
    distances = np.full(len(reference), np.inf)
    valid_right = indices < len(candidate)
    distances[valid_right] = np.minimum(
        distances[valid_right],
        np.abs(reference[valid_right] - candidate[indices[valid_right]]),
    )
    valid_left = indices > 0
    distances[valid_left] = np.minimum(
        distances[valid_left],
        np.abs(reference[valid_left] - candidate[indices[valid_left] - 1]),
    )
    return float(np.mean(distances <= tolerance_s))


def analyze_mouse(
    mouse: int,
    data_dir: Path,
    output_dir: Path,
    model_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if mouse == 12:
        raise ValueError("Mouse 12 is excluded because it duplicates mouse 11.")
    bundle = joblib.load(model_path)
    model = bundle["model"]
    threshold = float(bundle["threshold"])
    event_dir = output_dir / f"mouse_{mouse}_events"
    event_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for area in AREAS:
        path = data_dir / str(mouse) / f"lfp_{area}.npy"
        metadata = npy_metadata(path)
        channels = selected_channels(metadata[0][0])
        traces = load_npy_channels(path, channels, metadata)
        duration_min = metadata[0][1] / FS / 60
        for channel, trace in zip(channels, traces):
            print(f"Mouse {mouse}, area {area}, channel {channel}", flush=True)
            rms_events = detect_rms_events(trace)
            candidates = target_candidate_features(trace)
            if len(candidates):
                candidates["ml_probability"] = model.predict_proba(
                    candidates.loc[:, bundle["features"]]
                )[:, 1]
                accepted = candidates[candidates["ml_probability"] >= threshold].copy()
            else:
                candidates["ml_probability"] = pd.Series(dtype=float)
                accepted = candidates.copy()

            candidates.to_csv(
                event_dir / f"area_{area}_channel_{channel}_candidates.csv",
                index=False,
            )
            accepted.to_csv(
                event_dir / f"area_{area}_channel_{channel}_accepted.csv",
                index=False,
            )
            rms_events.to_csv(
                event_dir / f"area_{area}_channel_{channel}_rms.csv",
                index=False,
            )
            overlap = matched_fraction(
                rms_events.get("center_s", pd.Series(dtype=float)),
                accepted.get("center_s", pd.Series(dtype=float)),
            )
            rows.append(
                {
                    "mouse": mouse,
                    "brain_area": area,
                    "channel": int(channel),
                    "duration_min": duration_min,
                    "rms_events": len(rms_events),
                    "rms_events_per_min": len(rms_events) / duration_min,
                    "ml_candidates": len(candidates),
                    "ml_events": len(accepted),
                    "ml_events_per_min": len(accepted) / duration_min,
                    "fraction_rms_matched_by_ml": overlap,
                    "source_probability_threshold": threshold,
                }
            )

    channel_results = pd.DataFrame(rows)
    area_results = (
        channel_results.groupby(["mouse", "brain_area"], as_index=False)
        .agg(
            rms_events_per_min=("rms_events_per_min", "median"),
            ml_events_per_min=("ml_events_per_min", "median"),
            fraction_rms_matched_by_ml=("fraction_rms_matched_by_ml", "median"),
            n_channels=("channel", "count"),
        )
    )
    channel_results.to_csv(
        output_dir / f"mouse_{mouse}_channel_comparison.csv",
        index=False,
    )
    area_results.to_csv(
        output_dir / f"mouse_{mouse}_area_comparison.csv",
        index=False,
    )
    return channel_results, area_results


def _load_event_table(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["start_s", "stop_s", "center_s"])
    return pd.read_csv(path)


def plot_mouse_example(
    mouse: int,
    data_dir: Path,
    output_dir: Path,
    window_s: float = 3.0,
) -> Path:
    """Plot one channel per area with non-overlapping RMS/ML event rows."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    event_dir = output_dir / f"mouse_{mouse}_events"
    chosen = {}
    all_centers = []
    for area in AREAS:
        metadata = npy_metadata(data_dir / str(mouse) / f"lfp_{area}.npy")
        channel = int(selected_channels(metadata[0][0])[len(selected_channels(metadata[0][0])) // 2])
        rms = _load_event_table(
            event_dir / f"area_{area}_channel_{channel}_rms.csv"
        )
        ml = _load_event_table(
            event_dir / f"area_{area}_channel_{channel}_accepted.csv"
        )
        chosen[area] = {"channel": channel, "rms": rms, "ml": ml}
        for method, frame in (("rms", rms), ("ml", ml)):
            for center in frame.get("center_s", pd.Series(dtype=float)):
                all_centers.append((float(center), area, method))

    if not all_centers:
        raise RuntimeError("No Mouse 1 events are available for the example plot.")

    # Pick a compact window maximizing representation of both methods and areas.
    candidate_starts = [max(0.0, center - window_s / 2) for center, _, _ in all_centers]
    best_start = candidate_starts[0]
    best_score = (-1, -1)
    for start in candidate_starts:
        inside = [
            item for item in all_centers if start <= item[0] <= start + window_s
        ]
        coverage = len({(area, method) for _, area, method in inside})
        score = (coverage, len(inside))
        if score > best_score:
            best_score = score
            best_start = start
    best_stop = best_start + window_s

    colors = {"rms": "#0072B2", "ml": "#CC79A7"}
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    for area, ax in zip(AREAS, axes):
        channel = chosen[area]["channel"]
        path = data_dir / str(mouse) / f"lfp_{area}.npy"
        metadata = npy_metadata(path)
        pad_s = 1.0
        start_sample = max(0, int((best_start - pad_s) * FS))
        stop_sample = min(metadata[0][1], int((best_stop + pad_s) * FS))
        trace = load_npy_channels(path, np.array([channel]), metadata)[0]
        segment = np.asarray(trace[start_sample:stop_sample], dtype=float)
        sos = signal.butter(4, (100, 200), btype="bandpass", fs=FS, output="sos")
        filtered = signal.sosfiltfilt(sos, segment)
        segment_time = np.arange(start_sample, stop_sample) / FS
        display = robust_z(filtered)
        mask = (segment_time >= best_start) & (segment_time <= best_stop)
        ax.plot(
            segment_time[mask],
            display[mask],
            color="0.25",
            linewidth=0.7,
        )
        y_min, y_max = np.nanpercentile(display[mask], [1, 99])
        separation = max(0.8, 0.12 * (y_max - y_min))
        event_rows = {
            "rms": y_max + 1.2 * separation,
            "ml": y_max + 2.3 * separation,
        }
        for method in ("rms", "ml"):
            frame = chosen[area][method]
            for event in frame.itertuples(index=False):
                if event.stop_s < best_start or event.start_s > best_stop:
                    continue
                left = max(best_start, float(event.start_s))
                right = min(best_stop, float(event.stop_s))
                ax.hlines(
                    event_rows[method],
                    left,
                    right,
                    color=colors[method],
                    linewidth=5,
                    zorder=5,
                )
        ax.set_ylim(y_min - separation, y_max + 3.0 * separation)
        ax.set_ylabel("Robust z")
        ax.set_title(f"Area {area} — channel {channel}")
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("Session time (s)")
    handles = [
        Line2D([0], [0], color=colors["rms"], linewidth=5, label="Robust RMS"),
        Line2D([0], [0], color=colors["ml"], linewidth=5, label="Supervised ML"),
        Line2D([0], [0], color="0.25", linewidth=1, label="100–200 Hz LFP"),
    ]
    axes[0].legend(handles=handles, frameon=False, ncol=3, loc="upper right")
    fig.suptitle(
        f"Mouse {mouse}: ripple candidates in the same {window_s:g}-second window",
        y=0.995,
    )
    fig.tight_layout()
    output_path = output_dir / f"mouse_{mouse}_ripple_example.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("train", "mouse", "train-and-mouse"),
        default="train-and-mouse",
    )
    parser.add_argument("--mouse", type=int, default=1)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/q1_supervised"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/external_swr_cache"),
    )
    parser.add_argument("--sessions-per-dataset", type=int, default=3)
    parser.add_argument("--tables-per-session", type=int, default=1)
    parser.add_argument("--max-per-class-per-session", type=int, default=500)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "source_event_classifier.joblib"

    if args.mode in {"train", "train-and-mouse"}:
        start = time.perf_counter()
        train_source_model(
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            sessions_per_dataset=args.sessions_per_dataset,
            tables_per_session=args.tables_per_session,
            max_per_class_per_session=args.max_per_class_per_session,
        )
        print(f"Training wall time: {time.perf_counter() - start:.1f} s")
    if args.mode in {"mouse", "train-and-mouse"}:
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} does not exist. Run with --mode train first."
            )
        start = time.perf_counter()
        _, area_results = analyze_mouse(
            mouse=args.mouse,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            model_path=model_path,
        )
        inference_seconds = time.perf_counter() - start
        plot_path = plot_mouse_example(
            mouse=args.mouse,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
        )
        print("\nMouse-level comparison:")
        print(area_results.to_string(index=False))
        print(f"Inference wall time: {inference_seconds:.1f} s")
        print(
            f"Linear estimate for 17 mice: "
            f"{17 * inference_seconds / 60:.1f} min"
        )
        print(f"Example plot: {plot_path}")


if __name__ == "__main__":
    main()
