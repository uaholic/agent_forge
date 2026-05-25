from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from FlagEmbedding import BGEM3FlagModel
from langchain_huggingface import HuggingFaceEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "assets" / "models"

@lru_cache
def get_bge_base_model():
    return HuggingFaceEmbeddings(
        model_name=str(MODEL_DIR / "bge-base-zh-v1.5"),
    )

@lru_cache
def get_bge_m3_model():
    return BGEM3FlagModel(
        model_name_or_path=str(MODEL_DIR / "bge-m3"),
        use_fp16=False,
    )

def bge_base(content: str) -> list[float]:
    return get_bge_base_model().embed_query(content)


def bge_m3(sentences: List[str]) -> Dict:
    return get_bge_m3_model().encode(
        sentences=sentences,
        return_dense=True,  # 返回稠密向量
        return_sparse=True  # 返回稀疏向量
    )
