"""Check that the local Ollama server is responding and model is downloaded."""
import json
import os
import urllib.error
import urllib.request

url = os.getenv("OLLAMA_TAGS_URL", "http://localhost:11434/api/tags")
name = os.getenv("OLLAMA_MODEL", "qwen3:4b")
try:
    with urllib.request.urlopen(url, timeout=5) as response:
        result = json.load(response)
    installed = [m.get("name", "") for m in result.get("models", [])]
    print("Ollama server: OK")
    print("Installed models:", installed)
    if name not in installed:
        print(f"Install the requested model using: ollama pull {name}")
    else:
        print(f"Ready: {name}")
except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
    print(f"Cannot access Ollama server at {url}: {exc}")
    print("Start Ollama (or run 'ollama serve') and try again.")
    raise SystemExit(1)
