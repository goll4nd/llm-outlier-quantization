import torch
import torch.nn as nn


def _quant_blocks(w, bits, sym):
    """w: [out, n_groups, group] во float32. Возвращает деквантованный тензор."""
    if sym:
        qmax = 2 ** (bits - 1) - 1
        scale = (w.abs().amax(-1, keepdim=True) / qmax).clamp(min=1e-8)
        q = (w / scale).round().clamp(-qmax - 1, qmax)
        return q * scale
    else:
        qmax = 2 ** bits - 1
        mx = w.amax(-1, keepdim=True)
        mn = w.amin(-1, keepdim=True)
        scale = ((mx - mn) / qmax).clamp(min=1e-8)
        zero = (-mn / scale).round()
        q = ((w / scale) + zero).round().clamp(0, qmax)
        return (q - zero) * scale


def rtn_quantize_tensor(weight, bits=4, group_size=128, sym=False):
    """Fake-quantization весов матрицы [out_features, in_features].

    group_size: размер группы вдоль входной размерности.
                -1 = per-channel (одна группа на строку),
                 0 = per-tensor.
    """
    orig_dtype = weight.dtype
    w = weight.detach().float()
    out_f, in_f = w.shape

    if group_size == 0:                       # per-tensor
        flat = w.reshape(1, 1, -1)
        return _quant_blocks(flat, bits, sym).reshape(out_f, in_f).to(orig_dtype)

    g = in_f if group_size == -1 else group_size

    n_full = in_f // g
    rem = in_f - n_full * g

    parts = []
    if n_full > 0:
        head = w[:, :n_full * g].reshape(out_f, n_full, g)
        parts.append(_quant_blocks(head, bits, sym).reshape(out_f, n_full * g))
    if rem > 0:                               # неполная последняя группа
        tail = w[:, n_full * g:].reshape(out_f, 1, rem)
        parts.append(_quant_blocks(tail, bits, sym).reshape(out_f, rem))

    return torch.cat(parts, dim=1).to(orig_dtype)


def _is_quantizable(name, module):
    """Квантуем только Linear внутри трансформерных блоков."""
    if not isinstance(module, nn.Linear):
        return False
    if ".layers." not in name:                # отсекает lm_head и прочее вне блоков
        return False
    if "lm_head" in name or "embed" in name:
        return False
    return True


def quantize_model_rtn(model, bits=4, group_size=128, sym=False, verbose=False):
    """Квантует модель на месте. Возвращает статистику."""
    n_layers, n_params = 0, 0
    for name, module in model.named_modules():
        if not _is_quantizable(name, module):
            continue
        module.weight.data = rtn_quantize_tensor(
            module.weight.data, bits, group_size, sym
        )
        n_layers += 1
        n_params += module.weight.numel()
        if verbose:
            print(f"  {name}: {tuple(module.weight.shape)}")
    return {"layers": n_layers, "params": n_params}


def avg_bits_per_weight(bits, group_size, sym, in_features=4096):
    """Средняя разрядность с учётом накладных расходов на scale/zero."""
    if group_size == 0:
        return bits
    g = in_features if group_size == -1 else group_size
    overhead = 16 / g if sym else 32 / g      # scale (+ zero) в fp16
    return bits + overhead