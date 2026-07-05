import sys
from pathlib import Path
import numpy as np
from scipy import signal

def main():
    path = Path("data/1/lfp_1.npy")
    print("Checking file:", path)
    
    with path.open("rb") as file:
        version = np.lib.format.read_magic(file)
        print("Version:", version)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(file)
        else:
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(file)
        data_offset = file.tell()
        
    print(f"Shape: {shape}, Fortran: {fortran_order}, Dtype: {dtype}, Offset: {data_offset}")
    
    n_channels, n_samples = shape
    sos = signal.butter(4, (100.0, 200.0), btype="bandpass", fs=500.0, output="sos")
    
    # Try channel 0
    print("Mapping channel 0...")
    channel_offset = data_offset + 0 * n_samples * dtype.itemsize
    print("Offset:", channel_offset)
    
    try:
        trace = np.memmap(path, dtype=dtype, mode="r", offset=channel_offset, shape=(n_samples,))
        print("Mapped successfully. Trace shape:", trace.shape)
        trace_data = np.asarray(trace, dtype=float)
        print("Converted to float. Trace data shape:", trace_data.shape)
        trace._mmap.close()
        print("Closed mmap successfully.")
    except Exception as e:
        print("Error mapping channel 0:", e)
        return
        
    print("Filtering channel 0...")
    try:
        filtered = signal.sosfiltfilt(sos, trace_data)
        print("Filtered successfully. Var:", np.var(filtered))
    except Exception as e:
        print("Error filtering:", e)
        return
        
    print("All good!")

if __name__ == "__main__":
    main()
