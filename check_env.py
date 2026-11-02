import torch, platform
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

print("Python :", platform.python_version())
print("torch  :", torch.__version__)
print("CUDA   :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU    :", torch.cuda.get_device_name(0))
    print("VRAM   : %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1e9))

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print("\nЗагружаю OPT-125M...")
name = "facebook/opt-125m"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device)
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print("Параметров: %.1fM" % (n_params / 1e6))

ids = tok("The capital of France is", return_tensors="pt").to(device)
with torch.inference_mode():
    out = model.generate(**ids, max_new_tokens=15, do_sample=False)
print("Генерация:", tok.decode(out[0], skip_special_tokens=True))

print("\nСкачиваю WikiText-2...")
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
print("Строк в тесте:", len(ds))
print("Непустых:", sum(1 for t in ds["text"] if t.strip()))

print("\nВсё работает.")