"""Hermes-side mediator for the unilang runtime."""

from __future__ import annotations

import logging
import os
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from unilang import LanguageRuntime, SessionContext
    from unilang.config import LanguageMediationConfig

logger = logging.getLogger(__name__)


class UnilangMediator:
    """Thin Hermes wrapper around ``LanguageRuntime`` with safe pass-through defaults."""

    def __init__(self, language_mediation_config: Optional[Dict[str, Any]] = None) -> None:
        self._config = language_mediation_config or {}
        self._enabled = bool(self._config.get("enabled", False))
        self._runtime: Optional["LanguageRuntime"] = None
        self._runtime_config: Optional["LanguageMediationConfig"] = None
        self._session_ctx: Optional["SessionContext"] = None
        self._bound_session_id: Optional[str] = None
        self._last_assistant_provider_text: Optional[str] = None
        self._last_assistant_message_id: Optional[str] = None
        self._session_initialized = False

        if self._enabled:
            self._init_runtime()

    def bind_session(self, session_id: Optional[str]) -> None:
        if not session_id:
            return
        self._bound_session_id = session_id
        if self._session_ctx is None or self._session_ctx.session_id != session_id:
            self._session_ctx = None
        self._ensure_session_context(session_id=session_id)

    def _build_runtime_config(self):
        from unilang.config import (
            CompressionConfig,
            DelegationConfig,
            GatewayConfig,
            LanguageMediationConfig,
            MemoryConfig,
            OutputConfig,
            PromptArtifactConfig,
            ToolResultConfig,
            TranslatorConfig,
            TurnInputConfig,
        )

        adapter_cfg = self._config.get("adapter", {})
        return LanguageMediationConfig(
            enabled=self._enabled,
            provider_language=self._config.get("provider_language", "en"),
            translator=TranslatorConfig(
                provider=adapter_cfg.get("provider", "mock"),
                model=adapter_cfg.get("model", ""),
                base_url=adapter_cfg.get("base_url", ""),
                timeout_seconds=adapter_cfg.get("timeout_seconds", 30.0),
                failure_mode=adapter_cfg.get("failure_mode", "pass_through"),
            ),
            turn_input=TurnInputConfig(),
            output=OutputConfig(),
            prompt_artifacts=PromptArtifactConfig(enabled=False),
            tool_results=ToolResultConfig(enabled=True),
            compression=CompressionConfig(enabled=True),
            memory=MemoryConfig(enabled=True),
            delegation=DelegationConfig(),
            gateway=GatewayConfig(),
        )

    def _init_runtime(self) -> None:
        if self._runtime is not None:
            return
        try:
            from pathlib import Path

            from unilang import LanguageRuntime, PassthroughTranslationAdapter
            from unilang.content_classifier import ContentClassifier
            from unilang.language_cache import LanguageCache
            from unilang.language_detector import LanguageDetector
            from unilang.language_policy import LanguagePolicyEngine
            from unilang.prompt_artifacts import AllowAllPromptArtifactScanner
            from unilang.variant_store import VariantStore

            self._runtime_config = self._build_runtime_config()

            cache_cfg = self._config.get("cache", {})
            variant_cfg = self._config.get("variant_store", {})
            adapter_cfg = self._config.get("adapter", {})

            cache = None
            if cache_cfg.get("enabled", True):
                cache_path = cache_cfg.get("path") or str(Path.home() / ".hermes" / "unilang_cache.db")
                os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
                cache = LanguageCache(cache_path)

            variant_store = None
            if variant_cfg.get("enabled", False):
                variant_path = variant_cfg.get("path") or str(Path.home() / ".hermes" / "unilang_variants.db")
                os.makedirs(os.path.dirname(variant_path) or ".", exist_ok=True)
                variant_store = VariantStore(variant_path)

            adapter = self._build_adapter(adapter_cfg, PassthroughTranslationAdapter)
            detector = LanguageDetector(supported_languages=self._config.get("supported_languages"))

            self._runtime = LanguageRuntime(
                policy=LanguagePolicyEngine(),
                detector=detector,
                classifier=ContentClassifier(),
                adapter=adapter,
                cache=cache,
                variant_store=variant_store,
                prompt_artifact_scanner=AllowAllPromptArtifactScanner(),
            )
            logger.debug("UnilangMediator: LanguageRuntime initialised")
        except Exception as exc:
            logger.warning(
                "UnilangMediator: failed to initialise LanguageRuntime (%s). Falling back to pass-through mode.",
                exc,
            )
            self._enabled = False

    def _build_adapter(self, adapter_cfg: Dict[str, Any], passthrough_cls):
        provider = adapter_cfg.get("provider", "mock")
        if provider != "minimax":
            return passthrough_cls()

        api_key = adapter_cfg.get("api_key") or os.environ.get("MINIMAX_API_KEY", "")
        if not api_key:
            logger.warning(
                "UnilangMediator: MiniMax adapter selected but no API key found. Falling back to pass-through."
            )
            return passthrough_cls()

        try:
            from unilang.minimax_adapter import MiniMaxTranslationAdapter

            return MiniMaxTranslationAdapter(
                api_key=api_key,
                model=adapter_cfg.get("model", "MiniMax-M2.7-highspeed"),
                base_url=adapter_cfg.get("base_url", "https://api.minimax.io/anthropic"),
                timeout_seconds=adapter_cfg.get("timeout_seconds", 30.0),
                failure_mode=adapter_cfg.get("failure_mode", "pass_through"),
            )
        except ImportError:
            logger.warning(
                "UnilangMediator: MiniMax adapter requested but optional dependencies are missing. Falling back to pass-through."
            )
            return passthrough_cls()

    def _ensure_session_context(self, session_id: Optional[str] = None):
        if not self._enabled or self._runtime is None:
            return None

        from unilang import SessionContext

        if self._runtime_config is None:
            self._runtime_config = self._build_runtime_config()

        resolved_session_id = session_id or self._bound_session_id or f"unilang-{uuid.uuid4().hex[:12]}"
        self._bound_session_id = resolved_session_id
        if self._session_ctx is None or self._session_ctx.session_id != resolved_session_id:
            self._session_ctx = SessionContext(
                session_id=resolved_session_id,
                config=self._runtime_config,
            )
        self._session_initialized = True
        return self._session_ctx

    def _legacy_messages(self, messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        return messages, {"variant": "legacy", "mediated": False}

    def _legacy_text(self, messages: List[Dict[str, Any]]) -> str:
        return "\n".join(
            f"{message.get('role', '?')}: {message.get('content', '')}"
            for message in messages
            if isinstance(message.get("content"), str)
        )

    def normalize_input(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        *,
        session_id: Optional[str] = None,
    ) -> str:
        del conversation_history
        session_ctx = self._ensure_session_context(session_id=session_id)
        if session_ctx is None or self._runtime is None:
            return user_message
        try:
            normalized = self._runtime.normalize_user_message(session_ctx, user_message)
            return normalized.provider_text
        except Exception as exc:
            logger.warning("UnilangMediator.normalize_input: %s. Falling back to pass-through.", exc)
            return user_message

    def localize_output(
        self,
        assistant_content: str,
        turn_input: Optional[str] = None,
        *,
        session_id: Optional[str] = None,
    ) -> str:
        session_ctx = self._ensure_session_context(session_id=session_id)
        if session_ctx is None or self._runtime is None:
            return assistant_content
        try:
            if turn_input and not session_ctx.last_user_language:
                detection = self._runtime.detector.detect(turn_input)
                if detection.language_code:
                    session_ctx.last_user_language = detection.language_code
            localized = self._runtime.localize_assistant_output(session_ctx, assistant_content)
            self._last_assistant_provider_text = assistant_content
            if localized.render_variant is not None:
                self._last_assistant_message_id = localized.render_variant.message_id
            return localized.render_content
        except Exception as exc:
            logger.warning("UnilangMediator.localize_output: %s. Falling back to pass-through.", exc)
            return assistant_content

    def mediate_tool_result(
        self,
        tool_name: str,
        raw_content: str,
        *,
        session_id: Optional[str] = None,
    ) -> str:
        session_ctx = self._ensure_session_context(session_id=session_id)
        if session_ctx is None or self._runtime is None:
            return raw_content
        try:
            mediated = self._runtime.mediate_tool_result(session_ctx, raw_content, tool_name=tool_name)
            return mediated.provider_content
        except Exception as exc:
            logger.warning("UnilangMediator.mediate_tool_result: %s. Falling back to pass-through.", exc)
            return raw_content

    def prepare_prompt_artifacts(
        self,
        system_prompt: str,
        session_id: Optional[str] = None,
    ) -> str:
        session_ctx = self._ensure_session_context(session_id=session_id)
        if session_ctx is None or self._runtime is None:
            return system_prompt
        try:
            from unilang.types import PromptArtifact

            prepared = self._runtime.prepare_prompt_artifacts(
                session_ctx,
                [
                    PromptArtifact(
                        artifact_id="system-prompt",
                        kind="context_file",
                        content=system_prompt,
                        source_name="system_prompt",
                        allow_external_translation=True,
                    )
                ],
                force_rebuild=True,
            )
            if prepared.artifacts:
                return prepared.artifacts[0].prepared_text
            return system_prompt
        except Exception as exc:
            logger.warning("UnilangMediator.prepare_prompt_artifacts: %s. Falling back to pass-through.", exc)
            return system_prompt

    def prepare_compression_input(
        self,
        messages: List[Dict[str, Any]],
        selector: str = "provider",
        *,
        session_id: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        session_ctx = self._ensure_session_context(session_id=session_id)
        if session_ctx is None or self._runtime is None or self._runtime.variant_store is None:
            return self._legacy_messages(messages)
        try:
            compression_input = self._runtime.prepare_compression_input(session_ctx, selector=selector)
            mediated_messages = [
                {"role": message.role or "message", "content": message.content}
                for message in compression_input.messages
            ]
            return mediated_messages, {
                **compression_input.metadata,
                "variant": compression_input.selector_used,
                "mediated": True,
            }
        except Exception as exc:
            logger.warning("UnilangMediator.prepare_compression_input: %s. Falling back to legacy.", exc)
            return self._legacy_messages(messages)

    def persist_compression_summary(
        self,
        summary: str,
        original_messages: List[Dict[str, Any]],
        compressed_messages: List[Dict[str, Any]],
        *,
        session_id: Optional[str] = None,
    ) -> None:
        del original_messages, compressed_messages
        session_ctx = self._ensure_session_context(session_id=session_id)
        if session_ctx is None or self._runtime is None or self._runtime.variant_store is None:
            return
        try:
            self._runtime.persist_compression_summary(session_ctx, summary)
        except Exception as exc:
            logger.warning("UnilangMediator.persist_compression_summary: %s. Ignored.", exc)

    def prepare_memory_payload(
        self,
        messages: List[Dict[str, Any]],
        selector: str = "provider",
        *,
        session_id: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        session_ctx = self._ensure_session_context(session_id=session_id)
        if session_ctx is None or self._runtime is None or self._runtime.variant_store is None:
            return self._legacy_text(messages), {"variant": "legacy", "mediated": False}
        try:
            payload = self._runtime.prepare_memory_payload(
                session_ctx,
                path_kind="external",
                selector=selector,
            )
            return payload.content, {
                **payload.metadata,
                "variant": payload.selector_used or "summary",
                "mediated": True,
            }
        except Exception as exc:
            logger.warning("UnilangMediator.prepare_memory_payload: %s. Falling back to legacy.", exc)
            return self._legacy_text(messages), {"variant": "legacy", "mediated": False}

    def prepare_delegation_payload(
        self,
        child_session_context: Dict[str, Any],
        *,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        session_ctx = self._ensure_session_context(session_id=session_id)
        if session_ctx is None or self._runtime is None or self._runtime.variant_store is None:
            return child_session_context
        try:
            payload = self._runtime.prepare_delegation_payload(session_ctx)
            merged = dict(child_session_context)
            merged["language_mediation_payload"] = {
                "content": payload.content,
                "selector": payload.selector_used,
                "provider_language": payload.provider_language,
                "render_language": payload.render_language,
                "mediation_enabled": payload.mediation_enabled,
            }
            return merged
        except Exception as exc:
            logger.warning("UnilangMediator.prepare_delegation_payload: %s. Falling back to pass-through.", exc)
            return child_session_context

    def prepare_child_session_context(
        self,
        conversation_history: List[Dict[str, Any]],
        render_context: Optional[Dict[str, Any]] = None,
        *,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        session_ctx = self._ensure_session_context(session_id=session_id)
        if session_ctx is None or self._runtime is None:
            return {"conversation_history": conversation_history}
        try:
            child = self._runtime.build_child_session_context(
                session_ctx,
                child_session_id=f"{session_ctx.session_id}-child-{uuid.uuid4().hex[:8]}",
            )
            return {
                "conversation_history": conversation_history,
                "render_context": render_context,
                "language_mediation": {
                    "session_id": child.session_id,
                    "enabled": child.config.enabled,
                    "provider_language": child.config.provider_language,
                    "render_language": child.config.render_language,
                    "last_user_language": child.last_user_language,
                },
            }
        except Exception as exc:
            logger.warning("UnilangMediator.prepare_child_session_context: %s. Falling back to pass-through.", exc)
            return {"conversation_history": conversation_history}

    def prepare_gateway_message(
        self,
        surface: str = "render",
        *,
        session_id: Optional[str] = None,
    ) -> str:
        session_ctx = self._ensure_session_context(session_id=session_id)
        if (
            session_ctx is None
            or self._runtime is None
            or self._last_assistant_provider_text is None
        ):
            return ""
        try:
            gateway_message = self._runtime.prepare_gateway_message(
                session_ctx,
                self._last_assistant_provider_text,
                message_id=self._last_assistant_message_id,
                surface=surface,
            )
            return gateway_message.content
        except Exception as exc:
            logger.warning("UnilangMediator.prepare_gateway_message: %s. Returning empty string.", exc)
            return ""
