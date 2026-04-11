from __future__ import annotations

import sys
import types
from types import SimpleNamespace

from agent.unilang_mediator import UnilangMediator


class FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def normalize_user_message(self, session_ctx, raw_text):
        self.calls.append(("normalize", session_ctx.session_id, raw_text))
        session_ctx.last_user_language = "pt-BR"
        return SimpleNamespace(provider_text="normalized text")

    def localize_assistant_output(self, session_ctx, provider_text):
        self.calls.append(("localize", session_ctx.session_id, provider_text, session_ctx.last_user_language))
        return SimpleNamespace(
            render_content="texto localizado",
            render_variant=SimpleNamespace(message_id="assistant-1"),
        )

    def mediate_tool_result(self, session_ctx, raw_content, *, tool_name=None):
        self.calls.append(("tool", session_ctx.session_id, tool_name, raw_content))
        return SimpleNamespace(provider_content="tool output")

    def prepare_gateway_message(self, session_ctx, provider_text, *, message_id=None, surface=None):
        self.calls.append(("gateway", session_ctx.session_id, provider_text, message_id, surface))
        return SimpleNamespace(content=f"{surface}:{provider_text}")


def _build_enabled_mediator(fake_runtime: FakeRuntime) -> UnilangMediator:
    mediator = UnilangMediator({"enabled": False})
    mediator._enabled = True
    mediator._runtime = fake_runtime
    mediator._runtime_config = SimpleNamespace(enabled=True, provider_language="en")
    mediator._session_ctx = None
    return mediator


def test_build_runtime_config_maps_hermes_shape(monkeypatch):
    monkeypatch.setattr(UnilangMediator, "_init_runtime", lambda self: None)

    mediator = UnilangMediator(
        {
            "enabled": True,
            "provider_language": "en",
            "adapter": {
                "provider": "minimax",
                "model": "MiniMax-M2.7-highspeed",
                "base_url": "https://api.minimax.io/anthropic",
                "timeout_seconds": 12.5,
                "failure_mode": "pass_through",
            },
        }
    )

    runtime_config = mediator._build_runtime_config()

    assert runtime_config.enabled is True
    assert runtime_config.provider_language == "en"
    assert runtime_config.translator.provider == "minimax"
    assert runtime_config.translator.model == "MiniMax-M2.7-highspeed"
    assert runtime_config.tool_results.enabled is True
    assert runtime_config.compression.enabled is True
    assert runtime_config.memory.enabled is True


def test_build_adapter_uses_minimax_env_key(monkeypatch):
    mediator = UnilangMediator({"enabled": False})
    monkeypatch.setenv("MINIMAX_API_KEY", "env-key")

    class FakePassthrough:
        pass

    class FakeMiniMaxAdapter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(
        sys.modules,
        "unilang.minimax_adapter",
        types.SimpleNamespace(MiniMaxTranslationAdapter=FakeMiniMaxAdapter),
    )

    adapter = mediator._build_adapter({"provider": "minimax", "model": "mini"}, FakePassthrough)

    assert isinstance(adapter, FakeMiniMaxAdapter)
    assert adapter.kwargs["api_key"] == "env-key"
    assert adapter.kwargs["model"] == "mini"


def test_mediator_uses_runtime_contract_and_persists_session_language():
    fake_runtime = FakeRuntime()
    mediator = _build_enabled_mediator(fake_runtime)

    normalized = mediator.normalize_input("Oi", session_id="session-1")
    localized = mediator.localize_output("Hello", session_id="session-1")
    mediated_tool = mediator.mediate_tool_result("terminal", "resultado", session_id="session-1")
    gateway_text = mediator.prepare_gateway_message(surface="render", session_id="session-1")

    assert normalized == "normalized text"
    assert localized == "texto localizado"
    assert mediated_tool == "tool output"
    assert gateway_text == "render:Hello"
    assert fake_runtime.calls == [
        ("normalize", "session-1", "Oi"),
        ("localize", "session-1", "Hello", "pt-BR"),
        ("tool", "session-1", "terminal", "resultado"),
        ("gateway", "session-1", "Hello", "assistant-1", "render"),
    ]
