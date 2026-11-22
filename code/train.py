import numpy as np, torch, torch.nn as nn, os, sys, time, json

name = sys.argv[1]
epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 20
root = os.path.dirname(os.path.abspath(__file__))
dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
torch.manual_seed(0)
np.random.seed(0)
z = np.load(os.path.join(root, '..', 'data', 'sd.npz'))
x, mask, t, adj = z['x'], z['mask'], z['t'].astype('datetime64[m]'), z['adj']
n = x.shape[1]
bounds = {'train': ('2020-01-01', '2021-01-01'), 'val': ('2021-01-01', '2021-04-01'), 'test': ('2021-04-01', '2022-01-01')}
idx = {}
for s, (a, b) in bounds.items():
    lo, hi = np.searchsorted(t, np.datetime64(a)), np.searchsorted(t, np.datetime64(b))
    idx[s] = np.arange(lo + 7, hi - 4)
mu, sigma = x[:idx['train'][-1] + 5].mean(), x[:idx['train'][-1] + 5].std()
xn = torch.tensor((x - mu) / sigma, device=dev)
mk = torch.tensor(mask, device=dev)
a = adj - np.diag(np.diag(adj)) + np.eye(n, dtype=np.float32)
aFix = torch.tensor(a / a.sum(1, keepdims=True), device=dev)
d, de, alpha = 32, 10, 0.5
os.makedirs(os.path.join(root, 'ckpt'), exist_ok=True)
os.makedirs(os.path.join(root, 'results'), exist_ok=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.tc = nn.Sequential(nn.Conv1d(1, d, 3), nn.ReLU(), nn.Conv1d(d, d, 3), nn.ReLU())
        self.proj = nn.Linear(4 * d, d)
        self.wg = nn.Linear(d, d, bias=False)
        self.head = nn.Linear(d, 4)
        if name in ('adaptive', 'hybrid'):
            self.e1 = nn.Parameter(torch.randn(n, de))
            self.e2 = nn.Parameter(torch.randn(n, de))

    def forward(self, xb):
        b = xb.shape[0]
        zt = self.proj(self.tc(xb.permute(0, 2, 1).reshape(b * n, 1, 8)).reshape(b, n, 4 * d))
        if name == 'temporal': return self.head(zt).permute(0, 2, 1)
        g = aFix if name == 'fixed' else torch.softmax(torch.relu(self.e1 @ self.e2.T), 1)
        g = alpha * aFix + (1 - alpha) * g if name == 'hybrid' else g
        h = torch.relu(g @ self.wg(zt)) + zt
        return self.head(h).permute(0, 2, 1)

m = Net().to(dev)
opt = torch.optim.Adam(m.parameters(), 1e-3)
off = torch.arange(-7, 5, device=dev)

@torch.no_grad()
def evaluate(s, keep=False):
    m.eval()
    ae = se = cnt = 0
    preds = []
    for i in range(0, len(idx[s]), 128):
        w = torch.tensor(idx[s][i:i + 128], device=dev)[:, None] + off
        p = m(xn[w[:, :8]]) * sigma + mu
        y, mm = xn[w[:, 8:]] * sigma + mu, mk[w[:, 8:]]
        ae += ((p - y).abs() * mm).sum().item()
        se += (((p - y) ** 2) * mm).sum().item()
        cnt += mm.sum().item()
        if keep: preds.append(p.cpu().numpy())
    return ae / cnt, (se / cnt) ** 0.5, (np.concatenate(preds) if keep else None)

log, best, tStart = [], 1e9, time.time()
for ep in range(epochs):
    m.train()
    t0 = time.time()
    perm = torch.randperm(len(idx['train']))
    for i in range(0, len(perm), 64):
        w = torch.tensor(idx['train'][perm[i:i + 64].numpy()], device=dev)[:, None] + off
        mm = mk[w[:, 8:]]
        loss = ((m(xn[w[:, :8]]) - xn[w[:, 8:]]).abs() * mm).sum() / mm.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    sec = time.time() - t0
    vMae, vRmse, _ = evaluate('val')
    log.append(dict(epoch=ep + 1, loss=loss.item(), sec=sec, valMae=vMae, valRmse=vRmse))
    print(json.dumps(log[-1]), flush=True)
    if vMae < best:
        best = vMae
        torch.save(m.state_dict(), os.path.join(root, 'ckpt', f'{name}.pt'))
trainSec = time.time() - tStart
m.load_state_dict(torch.load(os.path.join(root, 'ckpt', f'{name}.pt')))
vMae, vRmse, _ = evaluate('val')
tMae, tRmse, pt = evaluate('test', True)
res = dict(name=name, epochs=epochs, bestEpoch=int(np.argmin([l['valMae'] for l in log]) + 1), trainSec=trainSec, secPerEpoch=float(np.mean([l['sec'] for l in log])), valMae=vMae, valRmse=vRmse, testMae=tMae, testRmse=tRmse, params=sum(p.numel() for p in m.parameters()), dev=str(dev), log=log)
json.dump(res, open(os.path.join(root, 'results', f'{name}.json'), 'w'), indent=1)
np.save(os.path.join(root, 'results', f'pred_{name}.npy'), pt[:, :, [0, 300, 600]])
if name in ('adaptive', 'hybrid'): np.save(os.path.join(root, 'results', f'adj_{name}.npy'), torch.softmax(torch.relu(m.e1 @ m.e2.T), 1).detach().cpu().numpy())
print(json.dumps({k: v for k, v in res.items() if k != 'log'}))
