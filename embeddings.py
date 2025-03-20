"""
This module initializes various embedding models.
"""
import os
from src.embedding import ZhipuEmbedding, OllamaEmbedding

glm_embedding_3 = ZhipuEmbedding(
    api_key=os.getenv("ZHIPU_API_KEY"),
    model="embedding-3",
)

mxbai_embed_large = OllamaEmbedding(
    host=os.getenv("OLLAMA_HOST"),
    model="mxbai-embed-large:latest",
)
