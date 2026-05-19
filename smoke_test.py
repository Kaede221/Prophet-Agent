"""Smoke test: verifies DeepSeek chat, SiliconFlow embedding, and Neo4j connectivity.

Run from project root:
    .venv/bin/python smoke_test.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from colorama import Fore, init as colorama_init

colorama_init()


def section(title):
    print(Fore.CYAN + f"\n=== {title} ===" + Fore.RESET)


def ok(msg):
    print(Fore.GREEN + "[OK] " + msg + Fore.RESET)


def fail(msg):
    print(Fore.RED + "[FAIL] " + msg + Fore.RESET)


def test_deepseek():
    section("DeepSeek chat (core.azure_gpt4.ask_gpt4o)")
    from core.azure_gpt4 import ask_gpt4o

    out = ask_gpt4o(
        system_prompt="You answer in strict JSON.",
        user_prompt='Return a JSON object with key "ping" and value "pong".',
        images=[],
        need_json=True,
    )
    print("Response:", out)
    assert isinstance(out, dict), f"Expected dict, got {type(out).__name__}: {out}"
    assert out.get("ping") == "pong", f"Unexpected payload: {out}"
    ok("DeepSeek returned valid JSON")


def test_embedding():
    section("SiliconFlow embedding (core.embedding.embeddings)")
    from core.embedding import embeddings

    vecs = embeddings(["hello world", "你好世界"])
    assert isinstance(vecs, list) and len(vecs) == 2, f"Bad shape: {len(vecs) if isinstance(vecs, list) else vecs}"
    assert all(isinstance(v, list) and len(v) > 0 for v in vecs), "Empty vectors"
    print(f"Got {len(vecs)} vectors, dim={len(vecs[0])}")
    ok("SiliconFlow embedding works")


def test_neo4j():
    section("Neo4j (py2neo)")
    import configparser
    from py2neo import Graph

    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(os.path.dirname(__file__), "config", "config.ini"), encoding="utf-8")
    uri = cfg.get("neo4j", "uri")
    user = cfg.get("neo4j", "user")
    password = cfg.get("neo4j", "password")

    g = Graph(uri, auth=(user, password))
    result = g.run("RETURN 1 AS x").data()
    assert result == [{"x": 1}], f"Unexpected: {result}"
    print("Server info:", g.service.connector.server_agent)
    ok("Neo4j reachable")


def main():
    failures = []
    for name, fn in [("deepseek", test_deepseek), ("embedding", test_embedding), ("neo4j", test_neo4j)]:
        try:
            fn()
        except Exception as e:
            fail(f"{name}: {e}")
            failures.append(name)

    section("Summary")
    if failures:
        fail("Failed: " + ", ".join(failures))
        sys.exit(1)
    ok("All smoke tests passed")


if __name__ == "__main__":
    main()
