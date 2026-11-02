"""Борьба с выбросами в активациях: SmoothQuant, QuaRot-подобные вращения, LLM.int8() (fake-quant)."""
from .quant import ActCfg, WCfg, QLinear, quantize_act, quantize_weight
from .pipeline import PipeCfg, prepare_model

MODELS = {
    "opt-125m": "facebook/opt-125m",
    "opt-350m": "facebook/opt-350m",
    "tinyllama": "TinyLlama/TinyLlama_v1.1",
    "phi3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "llama3-8b": "NousResearch/Meta-Llama-3-8B",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B",          # третье GLU-семейство; помещается целиком (3.1 ГБ FP16)
}
