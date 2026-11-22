import numpy as np, matplotlib, os, itertools
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')
os.makedirs(out, exist_ok=True)
K = 8
B = K + 1
phi = {-1: 1, 0: B, 1: B * B}
seen, ok = {}, True
for nm, n0, np1 in itertools.product(range(K + 1), repeat=3):
    if not 1 <= nm + n0 + np1 <= K: continue
    s = nm * phi[-1] + n0 * phi[0] + np1 * phi[1]
    ok &= (s % B, s // B % B, s // (B * B)) == (nm, n0, np1) and s not in seen
    seen[s] = (nm, n0, np1)
print('Q1b triples', len(seen), 'all decoded uniquely', ok)
print('Q1b non-steady count with phi(x)=x^2:', all(sum(x * x for x in nb) == sum(x != 0 for x in nb) for n in range(1, K + 1) for nb in itertools.product((-1, 0, 1), repeat=n)))
x = np.zeros(40)
x[20] = 1
y = np.roll(x, -1)
M = np.stack([x, np.roll(x, 1) + np.roll(x, -1)], 1)
a, b = np.linalg.lstsq(M, y, rcond=None)[0]
print('Q2a a, b =', round(a, 6), round(b, 6), 'SSE', ((M @ [a, b] - y) ** 2).sum())
i = np.arange(40)
x0 = np.exp(-(i - 20) ** 2 / 8)
pred, true = [x0], [x0]
for t in range(20):
    p = pred[-1]
    pred.append(a * p + b * (np.roll(p, 1) + np.roll(p, -1)))
    true.append(np.roll(true[-1], -1))
pred, true = np.array(pred), np.array(true)
fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for k, (m, tl) in enumerate(((pred, 'fitted layer (a=0, b=1/2)'), (true, 'true motion'))):
    im = ax[k].imshow(m, aspect='auto', origin='lower', cmap='viridis', vmin=0, vmax=1)
    ax[k].set(xlabel='sensor index', title=tl)
ax[0].set_ylabel('time step t')
fig.colorbar(im, ax=ax, label='slowdown')
fig.savefig(os.path.join(out, 'q2a.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(out, 'q2a.png'), dpi=120, bbox_inches='tight')
print('Q2a peak at t=20: pred', round(pred[-1].max(), 4), 'at', pred[-1].argmax(), '| true', round(true[-1].max(), 4), 'at', true[-1].argmax())
r = np.random.default_rng(0)
xb = r.random(40)
print('Q2b exact:', np.allclose(np.roll(xb, -1), 0 * xb + 0 * np.roll(xb, 1) + 1 * np.roll(xb, -1)))
xc = r.random(40)
back = np.allclose(np.roll(xc, -1), 0 * xc + (np.roll(xc, 1) + np.roll(xc, -1)) - np.roll(xc, 1))
fwd = np.allclose(np.roll(xc, 1), 0 * xc + (np.roll(xc, 1) + np.roll(xc, -1)) - np.roll(xc, -1))
print('Q2c backward exact:', back, '| forward exact:', fwd)
