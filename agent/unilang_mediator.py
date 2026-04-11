"""
unilang_mediator — Hermes-side Language Runtime mediator for unilang.

Wraps the host-agnostic LanguageRuntime and exposes a thin, Hermes-specific
API surface. All language mediation calls are gated behind ``enabled`` so that
English-only sessions are completely unaffected when the feature is disabled.

Integration seams
────────────────
1. Turn input  → ``normalize_input()``  before the user message is appended
2. Model output → ``localize_output()``  after the assistant message is built
3. Tool result  → ``mediate_tool_result()`` after each tool result is produced
4. Compression  → ``prepare_compression_input()`` / ``persist_compression_summary()``
5. Delegation   → ``prepare_delegation_payload()`` / ``prepare_child_context()``
6. Prompt artifacts → ``prepare_prompt_artifacts()`` on the system prompt

Safe by default
───────────────
``enabled=False`` means every method below is a zero-op (returns inputs unchanged,
no storage, no translation calls). Enable via ``language_mediation.enabled: true``
in ``~/.hermes/config.yaml`` or by passing ``language_mediation_config`` directly.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from unilang import LanguageRuntime
    from unilang.types import (
        DelegationPayload,
        GatewayMessage,
        PromptArtifact,
        ToolResultDecision,
    )

logger = logging.getLogger(__name__)


class UnilangMediator:
    """
    Thin, stateless mediator that delegates to ``LanguageRuntime``.

    Lives on ``AIAgent`` as ``self._unilang``.  All methods are no-ops
    when ``enabled=False``, so English-only deployments are completely unaffected.
    """

    def __init__(
        self,
        language_mediation_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            language_mediation_config: Top-level ``language_mediation`` dict from
                ``~/.hermes/config.yaml``.  When ``None`` or ``enabled=False``,
                all mediation methods become zero-ops.
        """
        self._config = language_mediation_config or {}
        self._enabled = bool(self._config.get("enabled", False))

        self._runtime: Optional["LanguageRuntime"] = None
        self._session_initialized = False

        if self._enabled:
            self._init_runtime()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _init_runtime(self) -> None:
        """Lazily initialise the LanguageRuntime once per session."""
        if self._runtime is not None:
            return
        try:
            from unilang import LanguageRuntime
            self._runtime = LanguageRuntime(config=self._config)
            logger.debug("UnilangMediator: LanguageRuntime initialised (enabled=True)")
        except Exception as exc:
            logger.warning(
                "UnilangMediator: failed to initialise LanguageRuntime (%s). "
                "Falling back to pass-through mode.",
                exc,
            )
            self._enabled = False

    def _init_session(self, session_id: str) -> None:
        """Called once per ``run_conversation`` session."""
        if not self._enabled or self._session_initialized:
            return
        if self._runtime is not None:
            try:
                self._runtime.init_session(session_id)
                self._session_initialized = True
            except Exception as exc:
                logger.warning(
                    "UnilangMediator: init_session failed (%s). Continuing in pass-through.",
                    exc,
                )

    # ── Turn input ───────────────────────────────────────────────────────────

    def normalize_input(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Normalize the user's input message to the provider's canonical language.

        Called BEFORE the user message is appended to the conversation history
        and before any API call.  Pass-through when disabled.

        Args:
            user_message: The raw user input string.
            conversation_history: Current conversation history (for context
                detection in the runtime).

        Returns:
            The normalised (translated) string, or the original if disabled.
        """
        self._init_session(None)
        if not self._enabled or self._runtime is None:
            return user_message
        try:
            return self._runtime.normalize_input(
                user_message,
                conversation_history=conversation_history,
            )
        except Exception as exc:
            logger.warning(
                "UnilangMediator.normalize_input: %s. Falling back to pass-through.",
                exc,
            )
            return user_message

    # ── Model output ──────────────────────────────────────────────────────────

    def localize_output(
        self,
        assistant_content: str,
        turn_input: Optional[str] = None,
    ) -> str:
        """
        Localise the assistant's response back to the user's language.

        Called after the assistant message is built but before it is persisted
        to the session DB and returned to the platform layer.  Pass-through
        when disabled.

        Args:
            assistant_content: The raw assistant response string.
            turn_input: Original user message from this turn (used for
                language detection if the runtime needs it).

        Returns:
            The localised string, or the original if disabled.
        """
        if not self._enabled or self._runtime is None:
            return assistant_content
        try:
            return self._runtime.localize_output(
                assistant_content,
                turn_input=turn_input,
            )
        except Exception as exc:
            logger.warning(
                "UnilangMediator.localize_output: %s. Falling back to pass-through.",
                exc,
            )
            return assistant_content

    # ── Tool results ──────────────────────────────────────────────────────────

    def mediate_tool_result(
        self,
        tool_name: str,
        raw_content: str,
    ) -> str:
        """
        Mediate a tool result before it is appended to the conversation.

        Tool results that contain natural-language summaries (long prose blocks)
        are translated to the provider language; code, paths, URLs, and
        structured data are preserved verbatim.  Pass-through when disabled.

        Args:
            tool_name: Name of the tool that produced this result.
            raw_content: The raw tool result string.

        Returns:
            The mediated tool result string.
        """
        if not self._enabled or self._runtime is None:
            return raw_content
        try:
            decision = self._runtime.mediate_tool_result(
                tool_name=tool_name,
                raw_content=raw_content,
            )
            if decision is None:
                return raw_content
            if isinstance(decision, dict):
                return decision.get("mediated_content", raw_content)
            return raw_content
        except Exception as exc:
            logger.warning(
                "UnilangMediator.mediate_tool_result: %s. Falling back to pass-through.",
                exc,
            )
            return raw_content

    # ── Prompt artifacts ───────────────────────────────────────────────────────

    def prepare_prompt_artifacts(
        self,
        system_prompt: str,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Scan and (if configured) translate prompt artifacts embedded in the
        system prompt.

        Args:
            system_prompt: The assembled system prompt string.
            session_id: Current session ID (used for cache scoping).

        Returns:
            The prompt with mediated artifact content, or the original if disabled.
        """
        if not self._enabled or self._runtime is None:
            return system_prompt
        try:
            return self._runtime.prepare_prompt_artifacts(
                system_prompt,
                session_id=session_id,
            )
        except Exception as exc:
            logger.warning(
                "UnilangMediator.prepare_prompt_artifacts: %s. Falling back to pass-through.",
                exc,
            )
            return system_prompt

    # ── Compression ────────────────────────────────────────────────────────────

    def prepare_compression_input(
        self,
        messages: List[Dict[str, Any]],
        selector: str = "provider",
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Prepare the conversation for context compression.

        Returns the canonical provider-language transcript for compression
        summarisation, along with metadata about what was selected.

        Args:
            messages: Current message list.
            selector: Which transcript variant to use (``provider`` | ``render``).

        Returns:
            A ``(compressed_messages, metadata)`` tuple where the messages
            are suitable for compression and ``metadata`` describes the
            selection (e.g., ``{"variant": "provider", "mediated": True}``).
        """
        if not self._enabled or self._runtime is None:
            return messages, {"variant": "legacy", "mediated": False}
        try:
            return self._runtime.prepare_compression_input(
                messages,
                selector=selector,
            )
        except Exception as exc:
            logger.warning(
                "UnilangMediator.prepare_compression_input: %s. Falling back to legacy.",
                exc,
            )
            return messages, {"variant": "legacy", "mediated": False}

    def persist_compression_summary(
        self,
        summary: str,
        original_messages: List[Dict[str, Any]],
        compressed_messages: List[Dict[str, Any]],
    ) -> None:
        """
        Persist a compression summary with variant tracking.

        Called after compression completes so that the runtime can store the
        provider-language canonical summary alongside any render variant.

        Args:
            summary: The compression summary text.
            original_messages: Messages before compression.
            compressed_messages: Messages after compression.
        """
        if not self._enabled or self._runtime is None:
            return
        try:
            self._runtime.persist_compression_summary(
                summary=summary,
                original_messages=original_messages,
                compressed_messages=compressed_messages,
            )
        except Exception as exc:
            logger.warning(
                "UnilangMediator.persist_compression_summary: %s. Ignored.",
                exc,
            )

    # ── Memory payload ────────────────────────────────────────────────────────

    def prepare_memory_payload(
        self,
        messages: List[Dict[str, Any]],
        selector: str = "provider",
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Prepare the transcript for external memory providers.

        Args:
            messages: Current message list.
            selector: Which variant to use (``provider`` | ``render``).

        Returns:
            A ``(transcript_text, metadata)`` tuple.
        """
        if not self._enabled or self._runtime is None:
            text = "\n".join(
                f"{m.get('role', '?')}: {m.get('content', '')}"
                for m in messages
                if isinstance(m.get("content"), str)
            )
            return text, {"variant": "legacy", "mediated": False}
        try:
            return self._runtime.prepare_memory_payload(
                messages,
                selector=selector,
            )
        except Exception as exc:
            logger.warning(
                "UnilangMediator.prepare_memory_payload: %s. Falling back to legacy.",
                exc,
            )
            legacy_text = "\n".join(
                f"{m.get('role', '?')}: {m.get('content', '')}"
                for m in messages
                if isinstance(m.get("content"), str)
            )
            return legacy_text, {"variant": "legacy", "mediated": False}

    # ── Delegation / subagent ────────────────────────────────────────────────

    def prepare_delegation_payload(
        self,
        child_session_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Prepare a delegation payload for a child (subagent) session.

        The child receives the provider-language canonical transcript by default.

        Args:
            child_session_context: The child session context dict to be
                forwarded to the subagent.

        Returns:
            The delegation payload with mediated content, or the original
            if disabled.
        """
        if not self._enabled or self._runtime is None:
            return child_session_context
        try:
            return self._runtime.prepare_delegation_payload(child_session_context)
        except Exception as exc:
            logger.warning(
                "UnilangMediator.prepare_delegation_payload: %s. Falling back to pass-through.",
                exc,
            )
            return child_session_context

    def prepare_child_session_context(
        self,
        conversation_history: List[Dict[str, Any]],
        render_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build a child session context for subagent spawning.

        Args:
            conversation_history: Parent session message history.
            render_context: Optional render-variant context from the parent.

        Returns:
            A child session context dict.
        """
        if not self._enabled or self._runtime is None:
            return {"conversation_history": conversation_history}
        try:
            return self._runtime.build_child_session_context(
                conversation_history=conversation_history,
                render_context=render_context,
            )
        except Exception as exc:
            logger.warning(
                "UnilangMediator.prepare_child_session_context: %s. Falling back to pass-through.",
                exc,
            )
            return {"conversation_history": conversation_history}

    # ── Gateway ───────────────────────────────────────────────────────────────

    def prepare_gateway_message(
        self,
        surface: str = "render",
    ) -> str:
        """
        Return the appropriate transcript variant for gateway delivery.

        Args:
            surface: Target surface (``render`` | ``provider``).

        Returns:
            The selected transcript as a string.
        """
        if not self._enabled or self._runtime is None:
            return ""
        try:
            return self._runtime.get_transcript(surface=surface)
        except Exception as exc:
            logger.warning(
                "UnilangMediator.prepare_gateway_message: %s. Returning empty string.",
                exc,
            )
            return ""
