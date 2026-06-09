from .videollama3_qwen2 import (
    Videollama3Qwen2Config,
    Videollama3Qwen2ForCausalLM,
)


VLLMs = {
    "videollama3_qwen2": Videollama3Qwen2ForCausalLM,
}

VLLMConfigs = {
    "videollama3_qwen2": Videollama3Qwen2Config,
}

__all__ = [
    "VLLMs",
    "VLLMConfigs",
    "Videollama3Qwen2Config",
    "Videollama3Qwen2ForCausalLM",
]
