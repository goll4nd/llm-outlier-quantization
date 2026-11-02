import random
from datasets import load_dataset


def get_calibration_samples(tokenizer, n_samples=128, seqlen=2048, seed=0):
    """Случайные отрывки из обучающего сплита WikiText-2."""
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(ds["text"])
    ids = tokenizer(text, return_tensors="pt").input_ids

    rng = random.Random(seed)
    samples = []
    for _ in range(n_samples):
        start = rng.randint(0, ids.shape[1] - seqlen - 1)
        chunk = ids[:, start:start + seqlen]
        samples.append({
            "input_ids": chunk[0],
            "attention_mask": chunk[0].new_ones(chunk.shape[1]),
        })
    return samples