import torch, time
from models import load_model
from evaluation.ppl import evaluate_perplexity

model, tok = load_model("opt-125m")
t0 = time.time()
ppl = evaluate_perplexity(model, tok)
print(f"\nOPT-125M FP16, WikiText-2: ppl = {ppl:.3f}")
print(f"Время: {time.time()-t0:.1f} c")
print(f"Пик VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} ГБ")