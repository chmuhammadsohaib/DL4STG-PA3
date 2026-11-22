import numpy as np, json, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = os.path.dirname(os.path.abspath(__file__))
rd, fd = os.path.join(root, 'results'), os.path.join(root, 'figs')
z = np.load(os.path.join(root, '..', 'data', 'sd.npz'))
x, t, adj, lat, lng = z['x'], z['t'].astype('datetime64[m]'), z['adj'], z['lat'], z['lng']
names = ['temporal', 'fixed', 'adaptive', 'hybrid']
res = {k: json.load(open(os.path.join(rd, f'{k}.json'))) for k in names}
for k in names: print(k, {c: round(res[k][c], 2) for c in ('valMae', 'valRmse', 'testMae', 'testRmse', 'trainSec', 'secPerEpoch')}, 'best epoch', res[k]['bestEpoch'], 'params', res[k]['params'])
lo = np.searchsorted(t, np.datetime64('2021-04-01')) + 7
wk = np.arange(np.searchsorted(t, np.datetime64('2021-04-05')), np.searchsorted(t, np.datetime64('2021-04-12'))) - lo
best = min(names[1:], key=lambda k: res[k]['testMae'])
fig, ax = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
for j, s in enumerate((0, 300, 600)):
    ax[j].plot(t[wk + lo + 4], x[wk + lo + 4, s], 'k', lw=1, label='actual')
    for k, c in (('temporal', 'tab:orange'), (best, 'tab:blue')): ax[j].plot(t[wk + lo + 4], np.load(os.path.join(rd, f'pred_{k}.npy'))[wk, 3, j], c, lw=0.8, label=f'{k} (1 h ahead)')
    ax[j].set(ylabel=f'sensor idx {s}\nveh / 15 min')
ax[0].legend(fontsize=8, ncol=3)
ax[2].set_xlabel('test week, Apr 5-11 2021')
fig.tight_layout()
fig.savefig(os.path.join(fd, 'q6_pred.pdf'))
la = np.load(os.path.join(rd, 'adj_adaptive.npy'))
n = la.shape[0]
off = la - np.diag(np.diag(la))
phys = (adj - np.diag(np.diag(adj))) != 0
top = np.argsort(-off, 1)[:, :10]
rows = np.repeat(np.arange(n), 10)
p1, p2 = np.radians(np.c_[lat, lng])[:, None], np.radians(np.c_[lat, lng])[None]
dist = 2 * 6371 * np.arcsin(np.sqrt(np.sin((p2[..., 0] - p1[..., 0]) / 2) ** 2 + np.cos(p1[..., 0]) * np.cos(p2[..., 0]) * np.sin((p2[..., 1] - p1[..., 1]) / 2) ** 2))
g = np.argsort(-off, None)[:100]
st = dict(top10InPhys=float(phys[rows, top.ravel()].mean()), physDensity=float(phys.mean()), top100InPhys=float(phys.ravel()[g].mean()), medDistTop10=float(np.median(dist[rows, top.ravel()])), medDistPhys=float(np.median(dist[phys])), medDistAll=float(np.median(dist[~np.eye(n, dtype=bool)])), fracTop10Within5km=float((dist[rows, top.ravel()] < 5).mean()), fracTop10Over20kmNotPhys=float(((dist[rows, top.ravel()] > 20) & ~phys[rows, top.ravel()]).mean()), selfWeightMean=float(np.diag(la).mean()), maxRowWeightMean=float(la.max(1).mean()))
json.dump(st, open(os.path.join(rd, 'q6.json'), 'w'), indent=1)
print(json.dumps(st))
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
im = ax[0].imshow(la, cmap='viridis', interpolation='nearest', vmax=np.percentile(la, 99.9))
ax[0].set(title='learned adjacency (adaptive)', xlabel='sensor j', ylabel='sensor i')
fig.colorbar(im, ax=ax[0], fraction=0.046)
ax[1].imshow(phys, cmap='Greys', interpolation='nearest')
ax[1].set(title='road-distance graph (nonzero)', xlabel='sensor j')
bins = np.linspace(0, 80, 41)
for v, lb in ((dist[rows, top.ravel()], 'learned top-10'), (dist[phys], 'road graph edges'), (dist[~np.eye(n, dtype=bool)], 'all pairs')): ax[2].hist(v, bins, density=True, histtype='step', lw=1.5, label=lb)
ax[2].set(title='distance between connected sensors', xlabel='km', ylabel='density')
ax[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(fd, 'q6_graph.pdf'))
