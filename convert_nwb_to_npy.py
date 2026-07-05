import os
import sys
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
import time

def convert_session(session_idx, source_base, dest_base):
    t_start = time.time()
    sub_name = f"sub-mouse-{session_idx}"
    nwb_name = f"{sub_name}_ses-None_ecephys.nwb"
    nwb_path = os.path.join(source_base, sub_name, nwb_name)
    
    print(f"\n==========================================")
    print(f"Processing Session {session_idx}...")
    print(f"Source file: {nwb_path}")
    
    if not os.path.exists(nwb_path):
        print(f"Error: Source file does not exist: {nwb_path}", file=sys.stderr)
        return False
        
    dest_dir = os.path.join(dest_base, "data", str(session_idx))
    
    # Check if all output files already exist to skip
    expected_files = [
        "brain_area.npy", "clusters.npy", "lfp_1.npy", "lfp_2.npy", "lfp_3.npy",
        "spikes.npy", "trial_data.csv", "trial_data.xlsx", "waveforms.npy"
    ]
    all_exist = True
    if os.path.exists(dest_dir):
        for f in expected_files:
            file_path = os.path.join(dest_dir, f)
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                all_exist = False
                break
    else:
        all_exist = False
        
    if all_exist:
        print(f"Session {session_idx} already converted. Skipping.")
        return True
        
    os.makedirs(dest_dir, exist_ok=True)
    print(f"Destination directory: {dest_dir}")
    
    try:
        print("Reading NWB file...")
        with NWBHDF5IO(nwb_path, 'r', load_namespaces=True) as io:
            nwbfile = io.read()
            
            # 1. Spikes & Clusters
            print("Processing spikes and clusters...")
            units_df = nwbfile.units.to_dataframe()
            all_spikes = []
            all_clusters = []
            for idx, row in units_df.iterrows():
                sp = row['spike_times']
                cid = row['cluster_id']
                all_spikes.append(sp)
                all_clusters.append(np.full(len(sp), cid))
                
            spikes = np.concatenate(all_spikes)
            clusters = np.concatenate(all_clusters)
            
            # Sort spikes and clusters chronologically
            sort_idx = np.argsort(spikes)
            spikes = spikes[sort_idx]
            clusters = clusters[sort_idx]
            
            spikes_file = os.path.join(dest_dir, "spikes.npy")
            clusters_file = os.path.join(dest_dir, "clusters.npy")
            np.save(spikes_file, spikes)
            np.save(clusters_file, clusters)
            print(f"  Saved spikes.npy (shape: {spikes.shape})")
            print(f"  Saved clusters.npy (shape: {clusters.shape})")
            
            # 2. Brain Areas mapping
            print("Processing brain areas...")
            cluster_ids = units_df['cluster_id'].values
            brain_areas = units_df['brain_area'].values
            brain_area_dict = {
                "cluster_id": cluster_ids,
                "brain_area": brain_areas
            }
            brain_area_file = os.path.join(dest_dir, "brain_area.npy")
            np.save(brain_area_file, brain_area_dict, allow_pickle=True)
            print(f"  Saved brain_area.npy (clusters: {len(cluster_ids)})")
            
            # 3. Waveforms
            print("Processing waveforms...")
            waveforms = np.stack(units_df['waveform_mean'].values)
            waveforms_file = os.path.join(dest_dir, "waveforms.npy")
            np.save(waveforms_file, waveforms)
            print(f"  Saved waveforms.npy (shape: {waveforms.shape})")
            
            # 4. Trials
            print("Processing trials...")
            trials_df = nwbfile.trials.to_dataframe()
            trials_out = pd.DataFrame()
            trials_out['variable_A'] = trials_df['is_variable_A']
            trials_out['variable_B'] = trials_df['is_variable_B']
            trials_out['variable_C'] = trials_df['variable_C']
            trials_out['trial_start'] = trials_df['start_time']
            trials_out['stim_start'] = trials_df['stim_start']
            trials_out['outcome'] = trials_df['outcome']
            trials_out['trial_end'] = trials_df['stop_time']
            trials_out['trial_duration'] = trials_out['trial_end'] - trials_out['trial_start']
            
            csv_file = os.path.join(dest_dir, "trial_data.csv")
            xlsx_file = os.path.join(dest_dir, "trial_data.xlsx")
            trials_out.to_csv(csv_file, index=True)
            trials_out.to_excel(xlsx_file, index=True)
            print(f"  Saved trial_data.csv and trial_data.xlsx (trials: {len(trials_out)})")
            
            # 5. LFPs
            print("Processing LFPs...")
            ecephys = nwbfile.processing['ecephys']
            lfp = ecephys.data_interfaces['LFP']
            
            # The shapes in NWB are (time, channels) -> Transpose to (channels, time)
            # Use [:] slicing for optimized h5py data loading
            lfp_1 = lfp.electrical_series['lfp_area_1'].data[:].T
            lfp_2 = lfp.electrical_series['lfp_area_2'].data[:].T
            lfp_3 = lfp.electrical_series['lfp_area_3'].data[:].T
            
            lfp_1_file = os.path.join(dest_dir, "lfp_1.npy")
            lfp_2_file = os.path.join(dest_dir, "lfp_2.npy")
            lfp_3_file = os.path.join(dest_dir, "lfp_3.npy")
            
            np.save(lfp_1_file, lfp_1)
            np.save(lfp_2_file, lfp_2)
            np.save(lfp_3_file, lfp_3)
            
            print(f"  Saved lfp_1.npy (shape: {lfp_1.shape})")
            print(f"  Saved lfp_2.npy (shape: {lfp_2.shape})")
            print(f"  Saved lfp_3.npy (shape: {lfp_3.shape})")
            
        print(f"Session {session_idx} completed in {time.time() - t_start:.2f} seconds.")
        return True
    except Exception as e:
        print(f"Error processing Session {session_idx}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False

def main():
    source_base = r"D:\218201"
    # Project directory is the current directory of the script
    dest_base = os.path.dirname(os.path.abspath(__file__))
    
    print(f"Starting conversion of 18 sessions...")
    print(f"Source: {source_base}")
    print(f"Destination: {dest_base}")
    
    success_count = 0
    t_total_start = time.time()
    
    for i in range(1, 19):
        if convert_session(i, source_base, dest_base):
            success_count += 1
            
    print(f"\n==========================================")
    print(f"Conversion complete. Successfully processed {success_count}/18 sessions.")
    print(f"Total time elapsed: {time.time() - t_total_start:.2f} seconds.")
    
    if success_count < 18:
        sys.exit(1)

if __name__ == "__main__":
    main()
