"""Stub environment variables so graph.py can be imported without infrastructure."""

import os

# Provide minimal stubs so graph.py can be imported without real infrastructure
os.environ.setdefault("HOST", "localhost")
os.environ.setdefault("USERNAME", "test")
os.environ.setdefault("PASSWORD", "test")
os.environ.setdefault("INDEX", "test")
os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("COHERE_API_KEY", "test")
os.environ.setdefault("ARIZE_SPACE_ID", "test")
os.environ.setdefault("ARIZE_API_KEY", "test")
