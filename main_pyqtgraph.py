import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import scipy as sp

dataset_folder = Path('data/')
mouse_id = 2
brain_area = 3 - 1
channel_id = 1

time_start = 0
time_end = 240
plot = False

fs = 500  # Hz, sampling rate
lfp_files = [f"lfp_1.npy", f"lfp_2.npy", f"lfp_3.npy"]
lfp_folder = dataset_folder / str(mouse_id)

# Loading the data
lfps = [np.load(lfp_folder / f) for f in lfp_files]

for i, lfp in enumerate(lfps, start=1):
    print(f"LFP{i}: shape {lfp.shape}, duration {lfp.shape[1]/fs:.1f} sec ({lfp.shape[1]/fs/60:.1f} min), "
          f"channels {lfp.shape[0]}")

x = lfps[brain_area][channel_id]
t = np.arange(0, len(x)) / fs
t_plot = np.arange(time_start*fs, time_end*fs) / fs

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

mean = abs(x_filtered).mean()
std = abs(x_filtered).std()
print(mean)
print(std)
threshold = std * 4
print(threshold)
threshold_signal = [threshold for n in range(len(t_plot))]

spikes = sp.signal.find_peaks(x_filtered, height=threshold, distance=200)
print("Number of spikes:", len(spikes[0]))

if plot:
    # ---- Plot Fourier transform (magnitude spectrum) ----
    plt.figure(figsize=(10,4))
    plt.plot(f, np.abs(X), label="|X(f)|")
    plt.xlim(0, 300)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Fourier Transform (Before Filtering)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if plot:
    # ---- Optional: plot filtered spectrum ----
    plt.figure(figsize=(10,4))
    plt.plot(f, np.abs(X), label="Original spectrum", alpha=0.5)
    plt.plot(f, np.abs(Xf), label="Filtered spectrum", linewidth=2)
    plt.xlim(0, 300)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Spectrum Before/After Filtering")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if plot or True:
    # Plot time-domain result using pyqtgraph: normalize both signals to [-1, 1] and trim first/last 1%
    try:
        from pyqtgraph.Qt import QtCore, QtWidgets
        import pyqtgraph as pg
    except Exception as e:
        print("pyqtgraph is required for interactive plotting. Install with: pip install pyqtgraph")
        raise

    x_seg = x[time_start*fs:time_end*fs]
    xf_seg = x_filtered[time_start*fs:time_end*fs]

    # trim first and last 1% to avoid edge artifacts
    n = len(x_seg)
    trim_start = int(0.01 * n)
    trim_end = int(0.99 * n)
    if trim_end <= trim_start:
        trim_start = 0
        trim_end = n

    t_plot_seg = t_plot[trim_start:trim_end]
    x_seg = x_seg[trim_start:trim_end]
    xf_seg = xf_seg[trim_start:trim_end]

    # normalization function (symmetric to preserve sign)
    def normalize(y):
        m = np.max(np.abs(y))
        return y / m if m != 0 else y

    x_norm = normalize(x_seg)
    xf_norm = normalize(xf_seg)

    # normalized threshold for plotting (recompute on trimmed filtered segment)
    max_abs_xf = np.max(np.abs(xf_seg)) if np.max(np.abs(xf_seg)) != 0 else 1.0
    threshold_norm = threshold / max_abs_xf
    threshold_signal = np.full_like(t_plot_seg, threshold_norm, dtype=float)

    # start (or reuse) Qt application
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    win = pg.GraphicsLayoutWidget(title="Time Domain Signal (normalized, trimmed 1%-99%)")
    win.resize(1000, 400)
    p = win.addPlot(title="Normalized signals")
    p.setLabel('bottom', 'Time', units='s')
    p.setLabel('left', 'Normalized amplitude')
    p.showGrid(x=True, y=True)

    p.plot(t_plot_seg, x_norm, pen=pg.mkPen('b', width=3), name='Original')
    p.plot(t_plot_seg, xf_norm, pen=pg.mkPen('r', width=0.25), name='Filtered')
    p.plot(t_plot_seg, threshold_signal, pen=pg.mkPen('g', width=1, style=QtCore.Qt.DashLine), name='Threshold')

    # mark spikes using normalized peak heights and times; only those within trimmed interval
    spike_times = spikes[0] / fs
    peak_heights = spikes[1]['peak_heights']
    t0, t1 = t_plot_seg[0], t_plot_seg[-1]
    spots = []
    for i, st in enumerate(spike_times):
        if st >= t0 and st <= t1:
            y_pos = peak_heights[i] / max_abs_xf
            spots.append({'pos': (st, y_pos), 'brush': pg.mkBrush(0, 255, 0), 'size': 8})
    if spots:
        scatter = pg.ScatterPlotItem(size=6, brush=pg.mkBrush(0, 255, 0))
        scatter.addPoints(spots)
        p.addItem(scatter)

    win.show()
    app.exec_()


