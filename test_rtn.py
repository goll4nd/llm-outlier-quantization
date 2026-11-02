import torch
from quantizers.rtn import rtn_quantize_tensor

torch.manual_seed(0)
w = torch.randn(64, 256)

# 1. При 16 битах искажения должны быть пренебрежимы
w16 = rtn_quantize_tensor(w, bits=16, group_size=128)
assert (w - w16).abs().max() < 1e-3, "16 бит не должны портить веса"

# 2. Уникальных значений в группе не больше 2^bits
for bits in (2, 3, 4, 8):
    wq = rtn_quantize_tensor(w, bits=bits, group_size=128)
    group = wq[0, :128]
    n_uniq = group.unique().numel()
    assert n_uniq <= 2 ** bits, f"{bits} бит: {n_uniq} уникальных значений"

# 3. Ошибка не превышает половину шага сетки
wq = rtn_quantize_tensor(w, bits=4, group_size=128)
g = w[0, :128]
step = (g.max() - g.min()) / 15
assert (w[0, :128] - wq[0, :128]).abs().max() <= step / 2 + 1e-5

# 4. Мельче группа — меньше ошибка
err = {gs: (w - rtn_quantize_tensor(w, 4, gs)).pow(2).mean().item()
       for gs in (-1, 256, 128, 64, 32)}
print("MSE по group_size:", {k: round(v, 6) for k, v in err.items()})
assert err[32] < err[64] < err[128] < err[-1]

# 5. Неполная группа не ломает форму
w_odd = torch.randn(32, 100)
assert rtn_quantize_tensor(w_odd, 4, 128).shape == w_odd.shape
assert rtn_quantize_tensor(w_odd, 4, 64).shape == w_odd.shape

print("Все проверки пройдены")