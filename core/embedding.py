import os
import configparser
from openai import OpenAI


def _load_config():
    config = configparser.ConfigParser()
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "config.ini"),
        "../config/config.ini",
        "config/config.ini",
    ]
    for path in candidates:
        if os.path.exists(path):
            config.read(path, encoding="utf-8")
            return config
    raise FileNotFoundError("config.ini not found in any of: " + ", ".join(candidates))


def embeddings(words):
    """Text embedding via an OpenAI-compatible endpoint (SiliconFlow by default)."""
    config = _load_config()
    key = config.get('gpt4', 'embedding_key')
    model = config.get('gpt4', 'embedding_model')
    base_url = config.get('gpt4', 'embedding_url')

    client = OpenAI(api_key=key, base_url=base_url)
    resp = client.embeddings.create(
        model=model,
        input=words,
        encoding_format="float",
    )
    return [item.embedding for item in resp.data]
