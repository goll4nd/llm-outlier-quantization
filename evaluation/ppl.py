import math
import torch
import torch.nn.functional as F
from datasets import load_dataset

_CACHE = {}

def get_wikitext2_ids(tokenizer):
    """Склеивает тестовый сплит в одну строку и токенизирует целиком."""
    key = tokenizer.name_or_path
    if key not in _CACHE:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"])
        _CACHE[key] = tokenizer(text, return_tensors="pt").input_ids
    return _CACHE[key]

@torch.inference_mode()
def evaluate_perplexity(model, tokenizer, seqlen=2048, device="cuda",
                        chunk=512, max_windows=None, verbose=True):
    ids = get_wikitext2_ids(tokenizer)
    n_windows = ids.numel() // seqlen
    if max_windows:
        n_windows = min(n_windows, max_windows)

    total_nll, total_tokens = 0.0, 0
    for i in range(n_windows):
        window = ids[:, i * seqlen:(i + 1) * seqlen].to(device)
        logits = model(window).logits

        shift_logits = logits[0, :-1, :]   # предсказания
        shift_labels = window[0, 1:]       # что должно было быть

        # считаем лосс кусками: float32-копия всего тензора логитов
        # для больших словарей (Llama-3) не влезет в память
        for s in range(0, shift_logits.size(0), chunk):
            lg = shift_logits[s:s + chunk].float()
            lb = shift_labels[s:s + chunk]
            total_nll += F.cross_entropy(lg, lb, reduction="sum").item()
            total_tokens += lb.numel()

        del logits, shift_logits
        if verbose and (i + 1) % 20 == 0:
            print(f"  окно {i+1}/{n_windows}, текущий ppl = "
                  f"{math.exp(total_nll / total_tokens):.3f}")

    return math.exp(total_nll / total_tokens)