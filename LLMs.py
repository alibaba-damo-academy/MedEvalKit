import os
from typing import Any

class LLMRegistry:
    _models = {}
    
    @classmethod
    def register(cls, name):
        def decorator(model_class):
            cls._models[name] = model_class
            return model_class
        return decorator
    
    @classmethod
    def get_model(cls, name):
        if name not in cls._models:
            raise ValueError(f"Model {name} not found in registry")
        return cls._models[name]

@LLMRegistry.register("Qwen2-VL")
class Qwen2VL:
    def __new__(cls, model_path: str, args: Any) -> Any:
        if os.environ.get("use_vllm", "True") == "True":
            from models.Qwen2_VL.Qwen2_VL_vllm import Qwen2VL
        else:
            from models.Qwen2_VL.Qwen2_VL_hf import Qwen2VL
        return Qwen2VL(model_path, args)

@LLMRegistry.register("Qwen2.5-VL") 
class Qwen2_5_VL:
    def __new__(cls, model_path: str, args: Any) -> Any:
        if os.environ.get("use_vllm", "True") == "True":
            from models.Qwen2_5_VL.Qwen2_5_VL_vllm import Qwen2_5_VL
        else:
            from models.Qwen2_5_VL.Qwen2_5_VL_hf import Qwen2_5_VL
        return Qwen2_5_VL(model_path, args)

@LLMRegistry.register("BiMediX2")
class BiMediX2:
    def __new__(cls, model_path: str, args: Any) -> Any:
        from models.BiMediX2.BiMediX2_hf import BiMediX2
        return BiMediX2(model_path, args)

@LLMRegistry.register("LLava_Med")
class LLavaMed:
    def __new__(cls, model_path: str, args: Any) -> Any:
        if os.environ.get("use_vllm", "True") == "True":
            from models.LLava_Med.LLava_Med_vllm import LLavaMed
        else:
            from models.LLava_Med.LLava_Med_hf import LLavaMed
        return LLavaMed(model_path, args)

@LLMRegistry.register("Huatuo")
class HuatuoGPT:
    def __new__(cls, model_path: str, args: Any) -> Any:
        if os.environ.get("use_vllm", "True") == "True":
            from models.HuatuoGPT.HuatuoGPT_vllm import HuatuoGPT
        else:
            from models.HuatuoGPT.HuatuoGPT_hf import HuatuoGPT
        return HuatuoGPT(model_path, args)

@LLMRegistry.register("InternVL")
class InternVL:
    def __new__(cls, model_path: str, args: Any) -> Any:
        if os.environ.get("use_vllm", "True") == "True":
            from models.InternVL.InternVL_vllm import InternVL
        else:
            from models.InternVL.InternVL_hf import InternVL
        return InternVL(model_path, args)

@LLMRegistry.register("Llama-3.2")
class LlamaVision:
    def __new__(cls, model_path: str, args: Any) -> Any:
        from models.Llama_3.Llama_3_2_vision_instruct_vllm import LlamaVision
        return LlamaVision(model_path, args)

@LLMRegistry.register("LLava")
class Llava:
    def __new__(cls, model_path: str, args: Any) -> Any:
        if os.environ.get("use_vllm", "True") == "True":
            from models.LLava.LLava_vllm import Llava
        else:
            from models.LLava.LLava_hf import Llava
        return Llava(model_path, args)


@LLMRegistry.register("Janus")
class Janus:
    def __new__(cls, model_path: str, args: Any) -> Any:
        from models.Janus.Janus import Janus
        return Janus(model_path, args)


@LLMRegistry.register("HealthGPT")
class HealthGPT:
    def __new__(cls, model_path: str, args: Any) -> Any:
        if "phi-4" in model_path:
            from models.HealthGPT.HealthGPT_phi import HealthGPT
        else:
            from models.HealthGPT.HealthGPT import HealthGPT
        return HealthGPT(model_path, args)

@LLMRegistry.register("BiomedGPT")
class BiomedGPT:
    def __new__(cls, model_path: str, args: Any) -> Any:
        from models.BiomedGPT.BiomedGPT import BiomedGPT
        return BiomedGPT(model_path, args)

@LLMRegistry.register("TestModel")
class TestModel:
    def __new__(cls, model_path: str, args: Any) -> Any:
        from models.TestModel.TestModel import TestModel
        return TestModel(model_path)

@LLMRegistry.register("Vllm_Text")
class VllmText:
    def __new__(cls, model_path: str, args: Any) -> Any:
        preprocessor_config_path = os.path.join(model_path, "preprocessor_config.json")
        if os.path.exists(preprocessor_config_path):
            from models.vllm_text.vllm_processor import Vllm_Text
        else:
            from models.vllm_text.vllm_tokenizer import Vllm_Text
        return Vllm_Text(model_path, args)

@LLMRegistry.register("MedGemma")
class MedGemma:
    def __new__(cls, model_path: str, args: Any) -> Any:
        if os.environ.get("use_vllm", "True") == "True":
            from models.MedGemma.MedGemma_vllm import MedGemma
        else:
            from models.MedGemma.MedGemma_hf import MedGemma
        return MedGemma(model_path, args)

@LLMRegistry.register("Med_Flamingo")
class MedFlamingo:
    def __new__(cls, model_path: str, args: Any) -> Any:
        from models.Med_Flamingo.Med_Flamingo_hf import Med_Flamingo
        return Med_Flamingo(model_path, args)
    
@LLMRegistry.register("GPT")
class GPT:
    def __new__(cls, model_path: str, args: Any) -> Any:
        from models.GPT.GPT import openai_llm
        return openai_llm(model_path, args)

@LLMRegistry.register("MedDr")
class MedDr:
    def __new__(cls, model_path: str, args: Any) -> Any:
        from models.MedDr.MedDr import MedDr
        return MedDr(model_path, args)

def init_llm(args):
    try:
        model_class = LLMRegistry.get_model(args.model_name)
        return model_class(args.model_path, args)
    except ValueError as e:
        raise ValueError(f"{args.model_name} not supported") from e

