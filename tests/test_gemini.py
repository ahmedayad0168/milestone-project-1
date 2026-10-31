import pytest

from src.generation import llm as llm_mod
from src.generation.llm import ExtractiveLLM, GeminiLLM, LLMUnavailable, build_llm

MSGS = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}, {"role": "user", "content": "q"}]


class Chunk:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def __init__(self, script):
        self.script, self.calls = script, []

    def generate_content_stream(self, model, contents, config):
        self.calls.append((model, config))
        action = self.script.pop(0)
        if isinstance(action, Exception):
            raise action
        return iter([Chunk(t) for t in action])


class FakeClient:
    def __init__(self, script):
        self.models = FakeModels(script)


def make(script, fallbacks=("backup",)):
    llm = GeminiLLM("key", "primary", list(fallbacks))
    llm._client = FakeClient(script)
    return llm


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)


def test_streams_text_and_maps_roles():
    llm = make([["Hel", "lo"]])
    assert "".join(llm.stream(MSGS)) == "Hello"
    model, cfg = llm._client.models.calls[0]
    assert model == "primary" and cfg.system_instruction == "sys"
    assert [c.role for c in llm._contents(MSGS)] == ["user", "model", "user"]


def test_retries_transient_error_then_succeeds():
    llm = make([RuntimeError("503 UNAVAILABLE"), ["ok"]])
    assert llm.chat(MSGS) == "ok" and len(llm._client.models.calls) == 2


def test_falls_back_to_next_model_on_hard_error():
    llm = make([RuntimeError("404 model not found"), ["from backup"]])
    assert llm.chat(MSGS) == "from backup"
    assert [m for m, _ in llm._client.models.calls] == ["primary", "backup"]


def test_retries_without_thinking_config_if_rejected():
    llm = make([RuntimeError("400 invalid argument: thinking_level"), ["fine"]])
    assert llm.chat(MSGS) == "fine"
    assert llm._client.models.calls[0][1].thinking_config is not None
    assert llm._client.models.calls[1][1].thinking_config is None


def test_raises_after_all_models_fail():
    llm = make([RuntimeError("404 nope"), RuntimeError("404 nope")])
    with pytest.raises(LLMUnavailable):
        llm.chat(MSGS)


def test_missing_key_degrades_to_extractive(monkeypatch):
    monkeypatch.setattr(llm_mod.settings, "LLM_BACKEND", "gemini")
    monkeypatch.setattr(llm_mod.settings, "GEMINI_API_KEY", "")
    assert isinstance(build_llm(), ExtractiveLLM)
