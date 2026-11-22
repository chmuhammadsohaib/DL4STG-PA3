# AI651 Assignment 3

## Setup
```bash
python3 -m venv .venv && .venv/bin/pip install torch numpy pandas h5py matplotlib kagglehub
```

## Data (LargeST, San Diego)
```bash
.venv/bin/python -c "import kagglehub; [kagglehub.dataset_download('liuxu77/largest', path=f, output_dir='data') for f in ['ca_meta.csv', 'ca_rn_adj.npy', 'ca_his_raw_2020.h5', 'ca_his_raw_2021.h5']]"
```

## Run
```bash
cd code
../.venv/bin/python q12.py              # Q1(b) check, Q2 heatmaps
../.venv/bin/python q3.py               # Q3: SD extraction, 15-min resampling, graph stats -> data/sd.npz
../.venv/bin/python train.py temporal 30
../.venv/bin/python train.py fixed 30
../.venv/bin/python train.py adaptive 30
../.venv/bin/python train.py hybrid 30
../.venv/bin/python q6.py               # Q6 table numbers, prediction plots, learned-graph analysis
```

## Configuration
- Input 8 steps (2 h), output 4 steps (1 h), 15-min data, 716 sensors.
- Splits: train 2020, val Jan-Mar 2021, test Apr-Dec 2021; normalisation from training data only.
- Model: temporal conv (Conv1d 1->32, k=3, ReLU, Conv1d 32->32, k=3, ReLU, Linear 128->32) -> graph conv H = ReLU(A Z W) + Z (skip), W 32x32 -> Linear 32->4. Temporal-only skips the graph conv.
- Graphs: fixed = row-normalised D^-1 (A + I); adaptive = softmax(relu(E1 E2^T)), E1, E2 in R^{716x10}; hybrid = 0.5 fixed + 0.5 adaptive.
- Training: Adam lr 1e-3, batch 64, 30 epochs, masked MAE loss, best validation epoch kept, seed 0.
- Hardware: Apple M3 Max (36 GB), PyTorch 2.15.0 on MPS.
