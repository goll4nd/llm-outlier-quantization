"""numpy-проверка алгебры, на которой построен torch-код (torch в песочнице недоступен)."""
import math
import numpy as np
from scipy.linalg import hadamard as sylvester

rng = np.random.default_rng(0)
ok = lambda name, c: print(("OK  " if c else "FAIL"), name)


# ---------- Адамар: Paley I / II, FWHT, кронекерово применение ----------
def is_prime(q):
    return q > 1 and all(q % d for d in range(2, int(q ** .5) + 1))

def jacobsthal(q):
    sq = {(i * i) % q for i in range(1, q)}
    chi = lambda a: 0 if a % q == 0 else (1 if a % q in sq else -1)
    return np.array([[chi(j - i) for j in range(q)] for i in range(q)], dtype=float)

def paley1(q):
    Q = jacobsthal(q); n = q + 1
    S = np.zeros((n, n)); S[0, 1:] = 1; S[1:, 0] = -1; S[1:, 1:] = Q
    return np.eye(n) + S

def paley2(q):
    Q = jacobsthal(q); n = q + 1
    C = np.zeros((n, n)); C[0, 1:] = 1; C[1:, 0] = 1; C[1:, 1:] = Q
    return np.kron(C, np.array([[1, -1], [-1, -1]])) + np.kron(np.eye(n), np.array([[1, 1], [1, -1]]))

for K, H in [(12, paley1(11)), (20, paley1(19)), (44, paley1(43)), (28, paley2(13)), (12, paley2(5)), (36, paley2(17))]:
    ok(f"Hadamard K={K}", np.allclose(H @ H.T, K * np.eye(K)) and set(np.unique(H)) <= {-1.0, 1.0})

def fwht(x):
    P = x.shape[-1]; lead = x.shape[:-1]; h = 1; y = x
    while h < P:
        y = y.reshape(*lead, P // (2 * h), 2, h)
        a, b = y[..., 0, :], y[..., 1, :]
        y = np.stack((a + b, a - b), axis=-2).reshape(*lead, P)
        h *= 2
    return y.reshape(x.shape) / math.sqrt(P)

P = 64; x = rng.standard_normal((3, P))
ok("FWHT == x @ H_sylvester/sqrt(P)", np.allclose(fwht(x), x @ sylvester(P) / math.sqrt(P)))

K = 12; HK = paley1(11) / math.sqrt(12); P = 16; n = K * P
x = rng.standard_normal((5, n))
Xr = x.reshape(5, K, P)
y = np.einsum('bkp,kc->bcp', fwht(Xr), HK).reshape(5, n)
Hfull = np.kron(HK, sylvester(P) / math.sqrt(P))
ok("kron application == x @ kron(H_K, H_P)", np.allclose(y, x @ Hfull))
ok("kron(H_K,H_P) orthogonal", np.allclose(Hfull @ Hfull.T, np.eye(n)))


# ---------- игрушечный pre-LN блок OPT / Llama и преобразования ----------
D, F, T = 24, 48, 7   # D = 12*2 -> Адамар 12 x Sylvester 2
nh, hd = 4, 6

def ln(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True); var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b

def rms(x, g=None, eps=1e-5):
    y = x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)
    return y if g is None else y * g

def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)

def lin(x, W, b=None):   # HF-соглашение: W [out, in]
    y = x @ W.T
    return y if b is None else y + b

def attn(x, P, rep=1):
    q = lin(x, P['q'], P.get('qb')); k = lin(x, P['k'], P.get('kb')); v = lin(x, P['v'], P.get('vb'))
    nkv = nh // rep
    q = q.reshape(T, nh, hd).transpose(1, 0, 2)
    k = np.repeat(k.reshape(T, nkv, hd).transpose(1, 0, 2), rep, 0)
    v = np.repeat(v.reshape(T, nkv, hd).transpose(1, 0, 2), rep, 0)
    mask = np.triu(np.full((T, T), -1e9), 1)
    o = softmax(q @ k.transpose(0, 2, 1) / math.sqrt(hd) + mask) @ v
    return lin(o.transpose(1, 0, 2).reshape(T, nh * hd), P['o'], P.get('ob'))

def opt_model(ids, M):
    h = M['E'][ids] + M['Epos'][np.arange(T)]
    for L in M['layers']:
        n1 = L['norm1'](h); h = h + attn(n1, L)
        n2 = L['norm2'](h); h = h + lin(np.maximum(lin(n2, L['fc1'], L['fc1b']), 0), L['fc2'], L['fc2b'])
    return lin(M['normf'](h), M['head'], M.get('headb'))

def mk_opt():
    r = lambda *s: rng.standard_normal(s) * 0.3
    M = {'E': r(50, D), 'Epos': r(T, D), 'head': None, 'layers': []}
    M['head'] = M['E'].copy()      # связанные эмбеддинги -> развязываем копией
    for _ in range(2):
        L = dict(q=r(D, D), k=r(D, D), v=r(D, D), o=r(D, D), qb=r(D), kb=r(D), vb=r(D), ob=r(D),
                 fc1=r(F, D), fc1b=r(F), fc2=r(D, F), fc2b=r(D))
        for nm in ('1', '2'):
            g, b = 1 + r(D), r(D); L['g' + nm], L['b' + nm] = g, b
        L['norm1'] = lambda x, L=L: ln(x, L['g1'], L['b1'])
        L['norm2'] = lambda x, L=L: ln(x, L['g2'], L['b2'])
        M['layers'].append(L)
    gf, bf = 1 + r(D), r(D); M['gf'], M['bf'] = gf, bf
    M['normf'] = lambda x: ln(x, M['gf'], M['bf'])
    return M

ids = rng.integers(0, 50, T)
M = mk_opt(); ref = opt_model(ids, M)

# --- SmoothQuant: LN -> qkv, LN -> fc1, плюс пары v->o и fc1->fc2 (ReLU) ---
import copy
S = copy.deepcopy(M)
for L in S['layers']:
    L['norm1'] = lambda x, L=L: ln(x, L['g1'], L['b1']); L['norm2'] = lambda x, L=L: ln(x, L['g2'], L['b2'])
    s = rng.uniform(0.2, 5, D)
    L['g1'] = L['g1'] / s; L['b1'] = L['b1'] / s
    for nm in 'qkv': L[nm] = L[nm] * s[None, :]
    s2 = rng.uniform(0.2, 5, D)
    L['g2'] = L['g2'] / s2; L['b2'] = L['b2'] / s2; L['fc1'] = L['fc1'] * s2[None, :]
    sv = rng.uniform(0.2, 5, D)                     # v -> o
    L['v'] = L['v'] / sv[:, None]; L['vb'] = L['vb'] / sv; L['o'] = L['o'] * sv[None, :]
    sf = rng.uniform(0.2, 5, F)                     # fc1 -> fc2 через ReLU (s > 0)
    L['fc1'] = L['fc1'] / sf[:, None]; L['fc1b'] = L['fc1b'] / sf; L['fc2'] = L['fc2'] * sf[None, :]
S['normf'] = lambda x: ln(x, S['gf'], S['bf'])
ok("SmoothQuant OPT (LN groups + v->o + fc1->fc2) invariant", np.allclose(opt_model(ids, S), ref, atol=1e-8))

# --- QuaRot для OPT: центрирование писателей, LN->RMSN, слияние gamma/beta, вращение Q, онлайн-H ---
R = copy.deepcopy(M)
Cn = lambda W: W - W.mean(0, keepdims=True)          # центрирование выхода писателя [D, in]
R['E'] = R['E'] - R['E'].mean(1, keepdims=True); R['Epos'] = R['Epos'] - R['Epos'].mean(1, keepdims=True)
for L in R['layers']:
    L['o'] = Cn(L['o']); L['ob'] = L['ob'] - L['ob'].mean()
    L['fc2'] = Cn(L['fc2']); L['fc2b'] = L['fc2b'] - L['fc2b'].mean()
    for nm in 'qkv':                                   # слияние LN1 в читателей
        L[nm + 'b'] = L[nm + 'b'] + L[nm] @ L['b1']; L[nm] = L[nm] * L['g1'][None, :]
    L['fc1b'] = L['fc1b'] + L['fc1'] @ L['b2']; L['fc1'] = L['fc1'] * L['g2'][None, :]
    L['norm1'] = rms; L['norm2'] = rms
R['headb'] = R['head'] @ R['bf']; R['head'] = R['head'] * R['gf'][None, :]; R['normf'] = rms
ok("OPT: center + LN->RMSN + fold (before rotation) invariant", np.allclose(opt_model(ids, R), ref, atol=1e-6))

signs = rng.choice([-1.0, 1.0], D)
Q = np.diag(signs) @ np.kron(paley1(11) / math.sqrt(12), sylvester(2) / math.sqrt(2))
ok("Q = D·H orthogonal", np.allclose(Q @ Q.T, np.eye(D)))
R['E'] = R['E'] @ Q; R['Epos'] = R['Epos'] @ Q; R['head'] = R['head'] @ Q
Hf = np.kron(paley2(5) / math.sqrt(12), sylvester(4) / math.sqrt(4))  # F = 48 = 12*4
for L in R['layers']:
    for nm in ('q', 'k', 'v', 'fc1'): L[nm] = L[nm] @ Q          # читатели: W Q
    L['o'] = Q.T @ L['o']; L['ob'] = L['ob'] @ Q                  # писатели: Q^T W, b Q
    L['fc2'] = Q.T @ L['fc2']; L['fc2b'] = L['fc2b'] @ Q
    L['fc2'] = L['fc2'] @ Hf                                      # онлайн-Адамар на входе fc2: x->xH, W->WH
    L['_hf'] = Hf
def opt_model_had(ids, M):
    h = M['E'][ids] + M['Epos'][np.arange(T)]
    for L in M['layers']:
        n1 = L['norm1'](h); h = h + attn(n1, L)
        n2 = L['norm2'](h); a = np.maximum(lin(n2, L['fc1'], L['fc1b']), 0) @ L['_hf']
        h = h + lin(a, L['fc2'], L['fc2b'])
    return lin(M['normf'](h), M['head'], M.get('headb'))
ok("QuaRot OPT (rotation + online H on fc2) invariant", np.allclose(opt_model_had(ids, R), ref, atol=1e-6))

# --- Llama-подобный блок, GQA: RMSNorm fold + rotation + SmoothQuant v->o с повтором по группам ---
rep = 2; nkv = nh // rep
def llama_model(ids, M, had_down=None):
    h = M['E'][ids]
    for L in M['layers']:
        n1 = rms(h, L.get('g1')); h = h + attn(n1, L, rep)
        n2 = rms(h, L.get('g2')); g = lin(n2, L['gate']); u = lin(n2, L['up'])
        a = g / (1 + np.exp(-g)) * u
        if had_down is not None: a = a @ had_down
        h = h + lin(a, L['down'])
    return lin(rms(h, M.get('gf')), M['head'])
r = lambda *s: rng.standard_normal(s) * 0.3
Ml = {'E': r(50, D), 'head': r(50, D), 'gf': 1 + r(D), 'layers': []}
for _ in range(2):
    Ml['layers'].append(dict(q=r(nh * hd, D), k=r(nkv * hd, D), v=r(nkv * hd, D), o=r(D, nh * hd),
                             gate=r(F, D), up=r(F, D), down=r(D, F), g1=1 + r(D), g2=1 + r(D)))
refl = llama_model(ids, Ml)
Sl = copy.deepcopy(Ml)
for L in Sl['layers']:
    sv = rng.uniform(0.2, 5, nkv * hd)
    so = np.repeat(sv.reshape(nkv, 1, hd), rep, 1).reshape(nh * hd)  # kv-голова g -> головы g*rep..g*rep+rep-1
    L['v'] = L['v'] / sv[:, None]; L['o'] = L['o'] * so[None, :]
    su = rng.uniform(0.2, 5, F)                                       # up -> down (SwiGLU линейна по up)
    L['up'] = L['up'] / su[:, None]; L['down'] = L['down'] * su[None, :]
    s1 = rng.uniform(0.2, 5, D); L['g1'] = L['g1'] / s1
    for nm in 'qkv': L[nm] = L[nm] * s1[None, :]
ok("SmoothQuant Llama GQA (v->o repeat, up->down, norm->qkv) invariant", np.allclose(llama_model(ids, Sl), refl, atol=1e-8))

Rl = copy.deepcopy(Ml)
for L in Rl['layers']:
    for nm in 'qkv': L[nm] = L[nm] * L['g1'][None, :] @ Q
    for nm in ('gate', 'up'): L[nm] = L[nm] * L['g2'][None, :] @ Q
    L['g1'] = None; L['g2'] = None
    L['o'] = Q.T @ L['o']; L['down'] = Q.T @ L['down'] @ Hf
Rl['E'] = Rl['E'] @ Q; Rl['head'] = (Rl['head'] * Rl['gf'][None, :]) @ Q; Rl['gf'] = None
ok("QuaRot Llama (fold + rotation + online H on down) invariant", np.allclose(llama_model(ids, Rl, Hf), refl, atol=1e-6))

# послойный вариант: повернуть буфер на входе блока 0 и вернуть перед финальной нормой
def llama_layerwise(ids, M, Q, had_down):
    h = M['E'][ids] @ Q
    for L in M['layers']:
        n1 = rms(h, L.get('g1')); h = h + attn(n1, L, rep)
        n2 = rms(h, L.get('g2')); g = lin(n2, L['gate']); u = lin(n2, L['up'])
        h = h + lin((g / (1 + np.exp(-g)) * u) @ had_down, L['down'])
    return lin(rms(h @ Q.T, M['gf']), M['head'])
Rw = copy.deepcopy(Ml)
for L in Rw['layers']:
    for nm in 'qkv': L[nm] = L[nm] * L['g1'][None, :] @ Q
    for nm in ('gate', 'up'): L[nm] = L[nm] * L['g2'][None, :] @ Q
    L['g1'] = None; L['g2'] = None
    L['o'] = Q.T @ L['o']; L['down'] = Q.T @ L['down'] @ Hf
ok("Layerwise rotation (buffer @ Q, @ Q^T before final norm) invariant", np.allclose(llama_layerwise(ids, Rw, Q, Hf), refl, atol=1e-6))

# --- почему smoothing по LN бессмысленно перед полным вращением: оно стирается при слиянии ---
# (q-вход после слияния = RMSN(x) Q независимо от s) — проверка тривиальна, пропускаем.

# --- SmoothQuant при alpha=0.5 уравнивает поканальные максимумы ---
ax = rng.uniform(0.1, 100, 10); aw = rng.uniform(0.01, 1, 10); s = np.sqrt(ax / aw)
ok("alpha=0.5 equalizes max|X_j|/s_j == s_j*max|W_j|", np.allclose(ax / s, s * aw))
