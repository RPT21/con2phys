import numpy as np

for session in range(1, 19):
    path = f"/Volumes/MorseSSD/FENS_Hackathon/data/{session}/brain_area.npy"
    out  = f"/Volumes/MorseSSD/FENS_Hackathon/data/{session}/brain_area.npz"

    a = np.load(path, allow_pickle=True)
    d = a.item()

    np.savez(out, cluster_id=d["cluster_id"], brain_area=d["brain_area"])
    print(f"Session {session}: cluster_id {d['cluster_id'].shape}, brain_area {d['brain_area'].shape}")