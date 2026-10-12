DEFAULT_TEST_PATH = "data/mavis/test.jsonl"
DEFAULT_RETRIEVAL_DIR = "data/retrieval/results/gpt-4o_1"


def get_backend(lib):
    if lib == "openai":
        from models.openai import ParallelGPT as MODEL
    elif lib == "anthropic":
        from models.claude import ParallelClaude as MODEL
    elif lib == "qwen":
        from models.qwen2_5_vl import QwenVL2_5 as MODEL
    elif lib == "llava":
        from models.llava_onevision import LLaVAOneVision as MODEL
    elif lib == "internvl":
        from models.internvl2_5 import InternVL2_5 as MODEL
    else:
        raise ValueError(f"Unknown --lib {lib!r}; choose from openai, anthropic, qwen, internvl, llava")
    return MODEL
