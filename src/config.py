"""
All settings live here. Values come from the .env file,
with safe defaults so the project runs without one.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def _get_int(name, default):
    value = os.getenv(name, str(default))
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"Setting {name} must be a whole number, got '{value}'")


def _get_bool(name, default):
    value = os.getenv(name, str(default)).strip().lower()
    return value in ("1", "true", "yes", "on")


def get_settings():
    return {
        # LLM used by RAG
        "llm_provider": os.getenv("LLM_PROVIDER", "ollama"),
        "llm_model": os.getenv("LLM_MODEL", "gemma3:4b"),
        "ollama_host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),

        # LLM used by agents (needs tool-calling support)
        "agent_model": os.getenv("AGENT_MODEL", "qwen3:4b"),
        "agent_think": _get_bool("AGENT_THINK", True),
        "agent_max_steps": _get_int("AGENT_MAX_STEPS", 5),

        # Embeddings
        "embed_provider": os.getenv("EMBED_PROVIDER", "ollama"),
        "embed_model": os.getenv("EMBED_MODEL", "nomic-embed-text"),

        # Chunking
        "chunker": os.getenv("CHUNKER", "word"),
        "chunk_size": _get_int("CHUNK_SIZE", 150),
        "chunk_overlap": _get_int("CHUNK_OVERLAP", 30),

        # Storage, retrieval and generation
        "vector_store": os.getenv("VECTOR_STORE", "memory"),
        "reranker": os.getenv("RERANKER", "keyword"),
        "generator": os.getenv("GENERATOR", "grounded"),
    }
