"""Ортонормированные матрицы Адамара произвольного (поддерживаемого) порядка.

n = K * P, где P = 2^k (Сильвестр, применяется быстрым преобразованием FWHT за O(n log n)),
а K — порядок «нечётной» части, для которого строится матрица Пэли:
  * Paley I:  K = q + 1,     q простое, q ≡ 3 (mod 4)   (12, 20, 24, 44, ...)
  * Paley II: K = 2 (q + 1), q простое, q ≡ 1 (mod 4)   (12, 28, 36, 60, ...)
Покрывает размеры моделей работы: 768 = 12·64, 3072 = 12·256, 5632 = 44·128 (TinyLlama FFN),
14336 = 28·512 (Llama-3-8B FFN), 2048/4096/8192/1024 — чистый Сильвестр.

Полная матрица H = kron(H_K, H_P) не материализуется: для строки x длины n
    x H = (H_K^T  X  H_P).reshape(n),   X = x.reshape(K, P).
Проверено в numpy (verify_math.py): ортогональность и совпадение с явным kron.
Если подходящего K нет (например 11008 = 172·64 у Llama-2-7B), берётся
фиксированная случайная ортогональная матрица порядка K с предупреждением.
"""
from __future__ import annotations

import math
import warnings
from functools import lru_cache

import torch


def _is_prime(q: int) -> bool:
    return q > 1 and all(q % d for d in range(2, int(q ** 0.5) + 1))


def _jacobsthal(q: int) -> torch.Tensor:
    residues = {(i * i) % q for i in range(1, q)}
    idx = torch.arange(q)
    diff = (idx[None, :] - idx[:, None]) % q
    chi = torch.zeros(q, dtype=torch.float64)
    for a in range(1, q):
        chi[a] = 1.0 if a in residues else -1.0
    return chi[diff]


def _paley1(q: int) -> torch.Tensor:
    n = q + 1
    S = torch.zeros(n, n, dtype=torch.float64)
    S[0, 1:] = 1.0
    S[1:, 0] = -1.0
    S[1:, 1:] = _jacobsthal(q)
    return torch.eye(n, dtype=torch.float64) + S


def _paley2(q: int) -> torch.Tensor:
    n = q + 1
    C = torch.zeros(n, n, dtype=torch.float64)
    C[0, 1:] = 1.0
    C[1:, 0] = 1.0
    C[1:, 1:] = _jacobsthal(q)
    A = torch.tensor([[1.0, -1.0], [-1.0, -1.0]], dtype=torch.float64)
    B = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float64)
    return torch.kron(C, A) + torch.kron(torch.eye(n, dtype=torch.float64), B)


def _construct(K: int) -> torch.Tensor | None:
    """±1-матрица Адамара порядка K или None."""
    if K == 1:
        return torch.ones(1, 1, dtype=torch.float64)
    if _is_prime(K - 1) and (K - 1) % 4 == 3:
        return _paley1(K - 1)
    if K % 2 == 0 and _is_prime(K // 2 - 1) and (K // 2 - 1) % 4 == 1:
        return _paley2(K // 2 - 1)
    return None


@lru_cache(maxsize=None)
def _factor(n: int) -> tuple[int, torch.Tensor, bool]:
    """(K, H_K ортонормированная float64 на CPU, is_hadamard)."""
    m, k = n, 0
    while m % 2 == 0:
        m //= 2
        k += 1
    if m == 1:
        return 1, torch.ones(1, 1, dtype=torch.float64), True
    for j in range(2, k + 1):           # K = m·2^j, K кратно 4
        K = m * 2 ** j
        H = _construct(K)
        if H is not None:
            assert torch.allclose(H @ H.T, K * torch.eye(K, dtype=torch.float64))
            return K, H / math.sqrt(K), True
    warnings.warn(f"Нет конструкции Адамара для n={n}: нечётная часть {m}; "
                  f"использую фиксированную случайную ортогональную матрицу порядка {m}.")
    g = torch.Generator().manual_seed(1234 + m)
    Qm, _ = torch.linalg.qr(torch.randn(m, m, generator=g, dtype=torch.float64))
    return m, Qm, False


_dev_cache: dict = {}


def _hk(n: int, device, dtype) -> tuple[int, torch.Tensor]:
    key = (n, str(device), dtype)
    if key not in _dev_cache:
        K, H, _ = _factor(n)
        _dev_cache[key] = (K, H.to(device=device, dtype=dtype))
    return _dev_cache[key]


def _fwht(x: torch.Tensor) -> torch.Tensor:
    """x @ H_P / sqrt(P) по последней оси (P = 2^k, порядок Сильвестра)."""
    P = x.shape[-1]
    lead = x.shape[:-1]
    h = 1
    y = x
    while h < P:
        y = y.reshape(*lead, P // (2 * h), 2, h)
        a, b = y[..., 0, :], y[..., 1, :]
        y = torch.stack((a + b, a - b), dim=-2).reshape(*lead, P)
        h *= 2
    return y / math.sqrt(P)


def hadamard_transform(x: torch.Tensor, compute_dtype=torch.float32) -> torch.Tensor:
    """x -> x H по последней оси, H ортонормированная (kron(H_K, H_P)). Возвращает dtype входа."""
    n = x.shape[-1]
    K, HK = _hk(n, x.device, compute_dtype)
    P = n // K
    y = x.to(compute_dtype).reshape(*x.shape[:-1], K, P)
    if P > 1:
        y = _fwht(y)
    if K > 1:
        y = torch.einsum("...kp,kc->...cp", y, HK)
    return y.reshape(x.shape).to(x.dtype)


def hadamard_matrix(n: int, dtype=torch.float64, device="cpu") -> torch.Tensor:
    """Явная ортонормированная H (строки = e_i H). Только для тестов и небольших n."""
    return hadamard_transform(torch.eye(n, dtype=dtype, device=device), compute_dtype=dtype)


def random_hadamard(n: int, seed: int = 0, dtype=torch.float64, device="cpu") -> torch.Tensor:
    """Q = diag(±1) H — рандомизированное преобразование Адамара (как в QuIP#/QuaRot)."""
    g = torch.Generator().manual_seed(seed)
    signs = torch.randint(0, 2, (n,), generator=g).to(dtype) * 2 - 1
    return hadamard_transform(torch.diag(signs).to(device), compute_dtype=dtype)


def random_orthogonal(n: int, seed: int = 0, dtype=torch.float64, device="cpu") -> torch.Tensor:
    """Случайная ортогональная матрица (QR гауссовой) — контроль «Адамар против случайного R»."""
    g = torch.Generator().manual_seed(seed)
    Q, R = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
    Q = Q * torch.sign(torch.diagonal(R))[None, :]   # равномерное распределение Хаара
    return Q.to(device=device, dtype=dtype)


def is_true_hadamard(n: int) -> bool:
    return _factor(n)[2]
