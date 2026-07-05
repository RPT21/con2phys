import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import scipy as sp

# iterate all mouse folders inside data/
dataset_folder = Path('data/')
lfp_files = ["lfp_1.npy", "lfp_2.npy", "lfp_3.npy"]
fs = 500  # Hz, sampling rate

output_folder = Path('data/output')
output_folder.mkdir(parents=True, exist_ok=True)

# find mouse folders (directories) inside data/
mouse_folders = sorted([p for p in dataset_folder.iterdir() if p.is_dir()])
if not mouse_folders:
    print(f'No mouse folders found in {dataset_folder.resolve()}')

for mouse_folder in mouse_folders:
    mouse_id = mouse_folder.name
    print(f"Processing mouse: {mouse_id} (folder: {mouse_folder})")

    # load lfps for this mouse
    try:
        lfps = [np.load(mouse_folder / f) for f in lfp_files]
    except Exception as e:
        print(f"Failed to load LFP files for mouse {mouse_id}: {e}")
        continue

    counts_all = []
    for i, lfp in enumerate(lfps, start=1):
        print(f"LFP{i}: shape {lfp.shape}, duration {lfp.shape[1]/fs:.1f} sec ({lfp.shape[1]/fs/60:.1f} min), channels {lfp.shape[0]}")

        counts = []
        for n in range(lfp.shape[0]):
            x = lfp[n]

            # 1) Fourier transform
            X = np.fft.rfft(x)
            f = np.fft.rfftfreq(len(x), d=1/fs)

            # 2) Frequency-domain filter (band-pass: 100-200 Hz)
            low_cutoff = 100
            high_cutoff = 200
            H = ((f >= low_cutoff) & (f <= high_cutoff)).astype(float)
            Xf = X * H

            # 3) Inverse transform to time domain
            x_filtered = np.fft.irfft(Xf, n=len(x))

            std = abs(x_filtered).std()
            threshold = std * 4

            spikes = sp.signal.find_peaks(x_filtered, height=threshold, width=2, distance=int(fs/10))
            count = len(spikes[0])
            counts.append(count)
            print(f"LFP{i}, Channel {n}: Number of spikes:", count)

        counts_all.append(counts)
        print(f'Counts for LFP{i}:', counts)

    # Plot all LFP histograms in a single figure with horizontal subplots and save
    n_lfps = len(counts_all)
    if n_lfps == 0:
        print('No LFPs found for this mouse.')
        continue

    fig, axes = plt.subplots(1, n_lfps, figsize=(5 * n_lfps, 4), sharey=True)
    if n_lfps == 1:
        axes = [axes]

    for idx, counts in enumerate(counts_all):
        ax = axes[idx]
        ax.bar(range(len(counts)), counts, color='tab:blue')
        ax.set_xlabel('Channel')
        if idx == 0:
            ax.set_ylabel('Spike count')
        ax.set_title(f'LFP{idx+1}')

    plt.tight_layout()
    out_path = output_folder / f"mouse_{mouse_id}_spike_counts.png"
    fig.savefig(out_path)
    plt.close(fig)

    print(f'Saved figure for mouse {mouse_id} to {out_path}')