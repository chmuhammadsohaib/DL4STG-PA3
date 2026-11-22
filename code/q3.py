import numpy as np, pandas as pd, h5py, os, json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = os.path.dirname(os.path.abspath(__file__))
dd, fd = os.path.join(root, '..', 'data'), os.path.join(root, 'figs')
os.makedirs(fd, exist_ok=True)
meta = pd.read_csv(os.path.join(dd, 'ca_meta.csv'))
sd = meta[meta.District == 11].reset_index(drop=True)
xs, ts = [], []
for y in (2020, 2021):
    f = h5py.File(os.path.join(dd, f'ca_his_raw_{y}.h5'), 'r')
    pos = {c.decode(): k for k, c in enumerate(f['t/axis0'][:])}
    sel = np.array([pos[str(i)] for i in sd.ID])
    v = f['t/block0_values']
    xs.append(np.concatenate([v[i:i + 8640][:, sel] for i in range(0, v.shape[0], 8640)]).astype(np.float32))
    ts.append(pd.to_datetime(f['t/axis1'][:]))
df = pd.DataFrame(np.concatenate(xs), index=ts[0].append(ts[1]))
x15 = df.resample('15min').mean() * 3
mask = ~x15.isna().values
x = x15.ffill().bfill().fillna(0).values.astype(np.float32)
adj = np.load(os.path.join(dd, 'ca_rn_adj.npy'))[np.ix_(sd.ID2.values, sd.ID2.values)].astype(np.float32)
np.savez_compressed(os.path.join(dd, 'sd.npz'), x=x, mask=mask, t=x15.index.values.astype('datetime64[m]').astype(np.int64), adj=adj, lat=sd.Lat.values, lng=sd.Lng.values)
off = adj - np.diag(np.diag(adj))
deg = (off != 0).sum(1)
s = int(np.argmax(deg == np.median(deg)))
hop1 = np.flatnonzero(off[s])
hop2 = np.setdiff1d(np.flatnonzero((off[hop1] != 0).any(0)), np.append(hop1, s))
st = dict(t5=len(df), t15=len(x15), miss5=float(df.isna().values.mean()), miss15=float(1 - mask.mean()), sensorsAllMissing=int((~mask).all(0).sum()), worstSensorMiss=float((1 - mask.mean(0)).max()), nodes=adj.shape[0], nnzAll=int((adj != 0).sum()), nnzOff=int((off != 0).sum()), diagNonzero=int((np.diag(adj) != 0).sum()), symmetric=bool(np.allclose(adj, adj.T)), asymPairs=int(((off != 0) != (off.T != 0)).sum() // 2), maxAbsAsym=float(np.abs(adj - adj.T).max()), wMin=float(off[off != 0].min()), wMax=float(off[off != 0].max()), degMean=float(deg.mean()), degMin=int(deg.min()), degMax=int(deg.max()), isolated=int((deg == 0).sum()), sensor=s, sensorId=int(sd.ID[s]), sensorFwy=sd.Fwy[s], hop1=[int(sd.ID[k]) for k in hop1], hop2=[int(sd.ID[k]) for k in hop2], nHop1=len(hop1), nHop2=len(hop2))
json.dump(st, open(os.path.join(root, 'q3.json'), 'w'), indent=1)
print(json.dumps(st))
wk = x15.loc['2020-01-06':'2020-01-12']
picks = [0, 300, 600]
fig, ax = plt.subplots(figsize=(11, 3.6))
for k in picks: ax.plot(wk.index, wk[k], lw=0.9, label=f'sensor {sd.ID[k]} ({sd.Fwy[k]})')
ax.set(xlabel='time (Jan 6-12, 2020)', ylabel='flow (vehicles / 15 min)')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(fd, 'q3_week.pdf'))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
im = ax[0].imshow(np.where(adj > 0, adj, np.nan), cmap='viridis', interpolation='nearest')
ax[0].set(title='SD road-distance adjacency (nonzero entries)', xlabel='sensor j', ylabel='sensor i')
fig.colorbar(im, ax=ax[0], fraction=0.046)
ax[1].hist(deg, bins=range(0, deg.max() + 2), color='tab:blue', edgecolor='k')
ax[1].set(title='number of neighbours per sensor', xlabel='neighbours (nonzero off-diagonal entries in row)', ylabel='sensors')
fig.tight_layout()
fig.savefig(os.path.join(fd, 'q3_adj.pdf'))
