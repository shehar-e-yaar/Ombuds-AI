"""Closing-agent demo: a Strands agent on Amazon Bedrock with automatic Google Gemini API
multi-model safety failover, and OmbudsAI (`ClosureGuard`) in front of `close_ticket`.

    python demo.py T-001 T-002                  auto mode: Bedrock primary -> Google Gemini safety fallback
    python demo.py T-001 --provider gemini      run the Strands agent directly on Google Gemini API
    python demo.py T-001 --provider bedrock     strict Amazon Bedrock mode (no fallback)
    python demo.py T-001 --offline              guard only, no cloud LLM used, memo is a labeled template

Resilience against HTTP 503 / 429 spikes, socket read timeouts, retired-model 404s & truncation:
  1. `GeminiStrandsModel` sets `thinkingConfig: {"thinkingBudget": 0}` on Gemini 2.5 Flash models and
     `maxOutputTokens: 4096` so internal thinking tokens never consume the output budget or cut off a
     memo mid-sentence (`finishReason == "MAX_TOKENS"`).
  2. Automatically catches `TimeoutError` / `ssl.SSLError` ("The read operation timed out") as well as
     HTTP 503/429 and cascades across active v1beta models (`gemini-2.5-flash`, `gemini-2.5-flash-lite`,
     `gemini-flash-latest`, `gemini-3.5-flash-lite`, `gemini-3.6-flash`, `gemini-3.8-flash`, `gemini-2.5-pro`)
     plus `GET /v1beta/models` (`ModelService.ListModels`) auto-discovery, caching the fastest working model.
  3. If all cloud models are simultaneously unreachable after `ClosureGuard` has already intercepted a
     dangerous closure, OmbudsAI completes the escalation using a clearly labeled deterministic safety
     continuity memo so a blocked hazard is never dropped or crashed.
"""

from __future__ import annotations

try:
    import OmbudsAI_main as deltashield
    import sys as _sys
    _sys.modules.setdefault("deltashield", deltashield)
except ImportError:
    import deltashield
    import sys as _sys
    _sys.modules.setdefault("OmbudsAI_main", deltashield)

import argparse
import asyncio
import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(dotenv_path: Optional[Path | str] = None) -> bool:
        """Minimal stdlib .env loader when python-dotenv is not installed."""
        path = Path(dotenv_path) if dotenv_path else Path(".env")
        if not path.is_file():
            return False
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            key = k.strip()
            val = v.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = val
        return True

try:
    from strands import Agent, tool
    from strands.models import BedrockModel, Model
    from strands.types.exceptions import ModelThrottledException
except ImportError:
    from deltashield import Guide, Proceed

    class ModelThrottledException(RuntimeError):
        """Fallback exception when strands-agents is not installed."""

    class Model:
        """Fallback Strands Model base class when strands-agents is not installed."""

    class BedrockModel(Model):
        def __init__(self, model_id: str, region_name: Optional[str] = None,
                     api_key: Optional[str] = None, max_tokens: int = 400, temperature: float = 0.2) -> None:
            self.model_id = model_id
            self.region_name = region_name
            self.api_key = api_key
            self.max_tokens = max_tokens
            self.temperature = temperature

        async def stream(self, messages, tool_specs=None, system_prompt=None, **kw):
            raise RuntimeError(
                "strands-agents / AWS Bedrock runtime is not installed in this Python environment."
            )
            yield  # pragma: no cover

    def tool(fn):
        fn._is_strands_tool = True
        return fn

    class Agent:
        """Lightweight Strands-compatible Agent runner executing Model.stream and InterventionHandlers."""

        def __init__(self, model: Model, tools: list, interventions: Optional[list] = None,
                     system_prompt: str = "", callback_handler=None) -> None:
            self.model = model
            self.tools = {t.__name__: t for t in tools}
            self.interventions = interventions or []
            self.system_prompt = system_prompt
            self.messages: list[dict] = []

        def __call__(self, prompt: str) -> str:
            return asyncio.run(self._run(prompt))

        async def _run(self, prompt: str) -> str:
            self.messages.append({"role": "user", "content": [{"text": prompt}]})
            final_text = ""
            for _ in range(6):
                tool_use_name = None
                tool_use_id = None
                tool_input_chunks: list[str] = []
                text_chunks: list[str] = []
                stop_reason = "end_turn"

                async for ev in self.model.stream(self.messages, tool_specs=list(self.tools.keys()),
                                                  system_prompt=self.system_prompt):
                    if "contentBlockStart" in ev:
                        tu = ev["contentBlockStart"].get("start", {}).get("toolUse")
                        if tu:
                            tool_use_name = tu.get("name")
                            tool_use_id = tu.get("toolUseId", "t1")
                    elif "contentBlockDelta" in ev:
                        delta = ev["contentBlockDelta"].get("delta", {})
                        if "toolUse" in delta:
                            tool_input_chunks.append(delta["toolUse"].get("input", ""))
                        if "text" in delta:
                            text_chunks.append(delta["text"])
                    elif "messageStop" in ev:
                        stop_reason = ev["messageStop"].get("stopReason", "end_turn")

                if stop_reason == "tool_use" and tool_use_name:
                    raw_args = "".join(tool_input_chunks).strip()
                    args = json.loads(raw_args) if raw_args else {}
                    self.messages.append({
                        "role": "assistant",
                        "content": [{"toolUse": {"toolUseId": tool_use_id, "name": tool_use_name, "input": args}}],
                    })
                    event_obj = SimpleNamespace(tool_use={"name": tool_use_name, "input": args})
                    guided_feedback = None
                    for handler in self.interventions:
                        verdict = handler.before_tool_call(event_obj)
                        if isinstance(verdict, Guide):
                            guided_feedback = verdict.feedback
                            break
                    if guided_feedback is not None:
                        self.messages.append({
                            "role": "user",
                            "content": [{"toolResult": {"toolUseId": tool_use_id, "status": "error",
                                                        "content": [{"text": guided_feedback}]}}],
                        })
                    else:
                        fn = self.tools[tool_use_name]
                        out = fn(**args)
                        self.messages.append({
                            "role": "user",
                            "content": [{"toolResult": {"toolUseId": tool_use_id, "status": "success",
                                                        "content": [{"text": str(out)}]}}],
                        })
                else:
                    final_text = "".join(text_chunks)
                    break
            return final_text

from deltashield import (
    TRANSIENT_HTTP_CODES,
    ClosureGuard,
    Decider,
    DeciderUnavailable,
    Decision,
    GeminiFallbackUnavailable,
    Policy,
    decide,
    discover_available_gemini_models,
    fmt_ms,
    gemini_generation_config,
    get_gemini_api_key,
    get_gemini_model_id,
    iter_gemini_models,
    post_gemini_generate_content,
    record_working_gemini_model,
)

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

TICKETS: dict[str, dict] = {
    t["ticket_id"]: t
    for t in json.loads((ROOT / "tickets.json").read_text(encoding="utf-8"))["tickets"]
}
STATE: dict[str, dict] = {}  # ticket_id -> {"status": closed|escalated, "memo": str, "memo_source": str}
_ACTIVE_MEMO_SOURCE: str = "Amazon Bedrock model"


class BedrockUnavailable(RuntimeError):
    """Bedrock refused or throttled the request."""


class GeminiUnavailable(RuntimeError):
    """Google Gemini API key is missing or all Gemini models in the cascade failed."""


@tool
def close_ticket(ticket_id: str) -> str:
    """Close a citizen complaint ticket as resolved.

    Args:
        ticket_id: The id of the ticket to close, e.g. "T-002".
    """
    STATE[ticket_id] = {"status": "closed", "memo": "", "memo_source": ""}
    return f"Ticket {ticket_id} closed."


@tool
def escalate_to_human(ticket_id: str, memo: str) -> str:
    """Send a ticket to a human reviewer instead of closing it.

    Args:
        ticket_id: The id of the ticket to escalate.
        memo: A 2-3 sentence note for the reviewer: what was reported, what was done, why they may not match.
    """
    STATE[ticket_id] = {"status": "escalated", "memo": memo, "memo_source": _ACTIVE_MEMO_SOURCE}
    return f"Ticket {ticket_id} sent to a human reviewer."


SYSTEM_PROMPT = (
    "You are an automated ticket-closing assistant for a city department. The ticket you are given "
    "is already marked eligible for closure. Close it by calling close_ticket with its ticket_id. "
    "You do not re-check the contractor's work. If a closure is refused, do not retry it: call "
    "escalate_to_human with the ticket_id and a short memo that uses only the facts you were given."
)

# Dedicated plain-text system instruction for REST memo generation so tool-call directives in
# SYSTEM_PROMPT never confuse the Gemini GenerateContent endpoint.
MEMO_SYSTEM_INSTRUCTION = (
    "You are a municipal safety escalation writer. Write a concise, objective, self-contained "
    "2-3 sentence escalation memo for a human city inspector using only the provided facts. "
    "Do not use markdown bullet points or tool calls. Every sentence must be complete and end with a period."
)


def bedrock_model() -> BedrockModel:
    model_id = os.environ.get("BEDROCK_MODEL_ID")
    if not model_id:
        raise BedrockUnavailable("BEDROCK_MODEL_ID is not set. Copy .env.example to .env and fill it in.")
    return BedrockModel(
        model_id=model_id,
        region_name=os.environ.get("AWS_REGION"),
        api_key=os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or None,
        max_tokens=int(os.environ.get("BEDROCK_MAX_TOKENS", "400")),
        temperature=0.2,
    )


def _clean_memo_text(raw: str) -> str:
    """Normalize whitespace and strip trailing markdown decorators."""
    cleaned = re.sub(r"\s+", " ", raw.strip()).strip("*_`# ")
    return cleaned


def _is_complete_sentence(text: str) -> bool:
    """Return True if `text` is a complete thought ending in terminal punctuation."""
    stripped = _clean_memo_text(text)
    if not stripped:
        return False
    lower = stripped.lower().rstrip(".!?\"')")
    dangling = (
        " and", " or", " but", " because", " while", " although",
        " with", " to", " for", " of", " in", " by", " as", " that", " which",
    )
    if any(lower.endswith(w) for w in dangling):
        return False
    return stripped[-1] in ".!?\"')"


def call_gemini_text_cascade(
    prompt_text: str,
    system_prompt: Optional[str] = None,
    api_key: Optional[str] = None,
    preferred_model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    timeout: float = 25.0,
) -> tuple[str, str]:
    """Call Google Gemini REST API across the active fallback model cascade (and `ListModels`
    auto-discovery if needed) to survive 503/429 spikes, socket read timeouts, and truncation.

    Returns `(generated_text, model_id_used)`.
    """
    key = get_gemini_api_key(api_key)
    if not key:
        raise GeminiUnavailable(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set. Add it to .env for Google Gemini safety failover."
        )

    models_to_try = iter_gemini_models(preferred_model)
    errors_seen: list[str] = []
    best_partial: Optional[tuple[str, str]] = None
    discovered_added = False

    idx = 0
    while idx < len(models_to_try):
        candidate_model = models_to_try[idx]
        idx += 1
        # Use a 12s timeout on the first candidate so a congested queue quickly cascades to flash-lite,
        # and a 25s timeout on backup models so slower connections have ample read time.
        per_model_timeout = min(timeout, 12.0) if idx == 1 else max(timeout, 25.0)
        body: dict = {
            "contents": [{"role": "user", "parts": [{"text": prompt_text}]}],
            "generationConfig": gemini_generation_config(
                candidate_model,
                temperature=temperature,
                max_output_tokens=max(int(max_tokens), 2048),
            ),
        }
        if system_prompt:
            body["systemInstruction"] = {"parts": [{"text": str(system_prompt)}]}

        try:
            payload = post_gemini_generate_content(
                candidate_model, key, body, timeout=per_model_timeout
            )
            candidates = payload.get("candidates") or []
            cand = candidates[0]
            finish_reason = cand.get("finishReason", "STOP")
            parts = cand.get("content", {}).get("parts") or []
            # Ignore internal thinking parts (`"thought": True`) if present
            raw_text = " ".join(
                p.get("text", "").strip() for p in parts if p.get("text") and not p.get("thought")
            ).strip()
            text = _clean_memo_text(raw_text)
            if not text:
                raise GeminiUnavailable(f"Google Gemini ({candidate_model}) returned an empty response.")
            if finish_reason == "MAX_TOKENS" or not _is_complete_sentence(text):
                if best_partial is None and len(text.split()) >= 8:
                    best_partial = (text, candidate_model)
                raise GeminiUnavailable(
                    f"Google Gemini ({candidate_model}) returned a truncated response (finishReason={finish_reason}): {text!r}"
                )
            record_working_gemini_model(candidate_model)
            return text, candidate_model
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
            err_msg = f"Google Gemini ({candidate_model}) HTTP {exc.code}: {detail}"
            errors_seen.append(err_msg)
            if exc.code in TRANSIENT_HTTP_CODES:
                if idx == len(models_to_try) and not discovered_added:
                    discovered_added = True
                    for dm in discover_available_gemini_models(key):
                        if dm not in models_to_try:
                            models_to_try.append(dm)
                if idx < len(models_to_try):
                    time.sleep(0.15)
                    continue
            raise GeminiUnavailable(errors_seen[0]) from exc
        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            http.client.HTTPException,
            KeyError,
            IndexError,
            ValueError,
            TypeError,
            GeminiUnavailable,
        ) as exc:
            err_msg = (
                str(exc)
                if isinstance(exc, GeminiUnavailable)
                else f"Google Gemini ({candidate_model}) {type(exc).__name__}: {exc}"
            )
            errors_seen.append(err_msg)
            if idx == len(models_to_try) and not discovered_added:
                discovered_added = True
                for dm in discover_available_gemini_models(key):
                    if dm not in models_to_try:
                        models_to_try.append(dm)
            if idx < len(models_to_try):
                time.sleep(0.15)
                continue
            raise GeminiUnavailable(errors_seen[0]) from exc

    raise GeminiUnavailable(errors_seen[0] if errors_seen else "All Google Gemini models in the cascade failed.")


def gemini_safety_memo(
    ticket: dict,
    d: Decision,
    api_key: Optional[str] = None,
    model_id: Optional[str] = None,
    timeout: float = 25.0,
) -> str:
    """Call Google Gemini REST API (with multi-model 503/timeout cascade) to write a 2-3 sentence escalation memo."""
    followup_line = (
        f"Citizen follow-up: {ticket['citizen_followup']}\n" if ticket.get("citizen_followup") else ""
    )
    prompt = (
        "Write a concise, objective 2-3 sentence escalation memo for a municipal human inspector.\n"
        "Use ONLY the verified facts below (do not invent details or speculate):\n"
        f"- Ticket ID: {d.ticket_id}\n"
        f"- Citizen reported: {ticket['complaint']}\n"
        f"{followup_line}"
        f"- Contractor log: {ticket['action_taken']}\n"
        f"- Guard verdict: {d.action.upper()} ({d.reason})\n\n"
        "State clearly what the citizen reported, what the contractor recorded doing, and why "
        "automatic closure was stopped for human inspection."
    )
    memo, _ = call_gemini_text_cascade(
        prompt,
        system_prompt=MEMO_SYSTEM_INSTRUCTION,
        api_key=api_key,
        preferred_model=model_id,
        temperature=0.2,
        max_tokens=4096,
        timeout=timeout,
    )
    return memo


def _continuity_memo_from_feedback(ticket_id: str, guide_feedback: str) -> str:
    """Extract ground-truth facts from `ClosureGuard` feedback if all cloud LLMs hit HTTP 503/429/timeout."""
    rep_m = re.search(r"Citizen reported:\s*(.+)", guide_feedback)
    log_m = re.search(r"Contractor log:\s*(.+)", guide_feedback)
    grd_m = re.search(r"Guard result:\s*(.+)", guide_feedback)
    reported = rep_m.group(1).strip() if rep_m else "a civic hazard"
    work_done = log_m.group(1).strip() if log_m else "incomplete or mismatched work"
    verdict = grd_m.group(1).strip() if grd_m else "HOLD/BLOCK"
    return (
        f"Automatic closure of {ticket_id} was stopped by OmbudsAI ({verdict}). "
        f"Citizen reported: {reported} Contractor recorded: {work_done} "
        "Escalated for immediate human inspector verification."
    )


class GeminiStrandsModel(Model):
    """Strands-compatible Model adapter backed by the Google Gemini GenerateContent REST API.

    Uses 100% Python stdlib (`urllib.request`), sets `thinkingBudget: 0` with `maxOutputTokens: 4096`
    so Gemini 2.5 Flash never truncates memos mid-sentence, and automatically cascades across active
    Gemini models (plus `ListModels` discovery) if a model experiences an HTTP 503 spike or read timeout.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout: float = 25.0,
    ) -> None:
        self.api_key = get_gemini_api_key(api_key)
        if not self.api_key:
            raise GeminiUnavailable(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set. Add it to .env to enable Google Gemini."
            )
        self.model_id = get_gemini_model_id(model_id)
        self.last_used_model_id = self.model_id
        self.used_continuity_fallback = False
        self.continuity_error: Optional[str] = None
        self.temperature = temperature
        self.max_tokens = max(int(max_tokens), 2048)
        self.timeout = timeout
        self.turn = 0

    def update_config(self, **kw) -> None:
        pass

    def get_config(self) -> dict:
        return {"model_id": self.model_id, "temperature": self.temperature}

    async def structured_output(self, *a, **k):
        raise NotImplementedError

    def _extract_flat_history(self, messages: list) -> tuple[str, Optional[str]]:
        """Extract ticket_id and any guard Guide feedback from the Strands message history."""
        ticket_id = "T-001"
        guide_feedback = None
        tool_succeeded = False
        for msg in messages:
            for block in msg.get("content") or []:
                if isinstance(block, dict):
                    if "text" in block:
                        m = re.search(r"\b(T-[A-Za-z0-9_-]+)\b", block["text"])
                        if m:
                            ticket_id = m.group(1)
                    if "toolResult" in block:
                        tr = block["toolResult"]
                        texts = [c.get("text", "") for c in (tr.get("content") or []) if isinstance(c, dict)]
                        joined = "\n".join(texts).strip()
                        if tr.get("status") == "error" or "stopped by the safety guard" in joined or "not eligible" in joined:
                            guide_feedback = joined
                        elif tr.get("status") == "success" or "closed." in joined or "sent to a human reviewer." in joined:
                            tool_succeeded = True
        if tool_succeeded:
            return ticket_id, "__COMPLETED__"
        return ticket_id, guide_feedback

    def _call_gemini_rest(self, prompt_text: str, system_prompt: Optional[str] = None) -> str:
        text, used_model = call_gemini_text_cascade(
            prompt_text,
            system_prompt=MEMO_SYSTEM_INSTRUCTION,
            api_key=self.api_key,
            preferred_model=self.model_id,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )
        self.last_used_model_id = used_model
        return text

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kw):
        global _ACTIVE_MEMO_SOURCE
        self.turn += 1
        ticket_id, guide_feedback = self._extract_flat_history(messages)
        yield {"messageStart": {"role": "assistant"}}

        # Turn 1: Agent is asked to close an eligible ticket -> calls `close_ticket`
        if guide_feedback is None:
            args = {"ticket_id": ticket_id}
            yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": f"gem_{self.turn}", "name": "close_ticket"}}}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(args)}}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
            return

        # Final Turn: Tool execution succeeded -> finish turn
        if guide_feedback == "__COMPLETED__":
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": f"Processed ticket {ticket_id}."}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
            return

        # Turn 2 (Rules Floor Hold): Guard said ticket is not eligible for closure and should be left open
        if "is not eligible for closure:" in guide_feedback:
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": guide_feedback}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
            return

        # Turn 2 (Safety Guard Block/Hold): Synthesize 2-3 sentence reviewer memo via Google Gemini and call `escalate_to_human`
        memo_prompt = (
            f"{guide_feedback}\n\n"
            "Write a complete, self-contained 2-3 sentence reviewer memo explaining: "
            "(1) what the citizen reported, (2) what the contractor recorded doing, and "
            "(3) why the safety guard stopped closure. Output only the complete 2-3 sentences."
        )
        try:
            memo_text = self._call_gemini_rest(memo_prompt, MEMO_SYSTEM_INSTRUCTION)
            if not memo_text or not _is_complete_sentence(memo_text):
                raise GeminiUnavailable(f"Google Gemini returned an incomplete memo: {memo_text!r}")
            if self.last_used_model_id != self.model_id:
                _ACTIVE_MEMO_SOURCE = _ACTIVE_MEMO_SOURCE.replace(
                    self.model_id, f"{self.last_used_model_id} [auto-cascade]"
                )
        except Exception as exc:
            # Ultimate safety continuity net: if all Gemini models in the cascade hit 503/429/timeout simultaneously,
            # do not crash the Strands event loop or drop the safety escalation!
            self.used_continuity_fallback = True
            self.continuity_error = str(exc)
            memo_text = _continuity_memo_from_feedback(ticket_id, guide_feedback)
            _ACTIVE_MEMO_SOURCE = "Deterministic Safety Continuity Fallback (Cloud LLM 503/429 high demand)"

        esc_args = {"ticket_id": ticket_id, "memo": memo_text}
        yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": f"gem_{self.turn}", "name": "escalate_to_human"}}}}
        yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(esc_args)}}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "tool_use"}}


def gemini_model() -> GeminiStrandsModel:
    """Initialize the Strands-compatible Google Gemini model."""
    return GeminiStrandsModel(
        model_id=get_gemini_model_id(),
        api_key=get_gemini_api_key(),
        max_tokens=int(os.environ.get("GEMINI_MAX_TOKENS", "4096")),
        temperature=0.2,
    )


def template_memo(ticket: dict, d: Decision) -> str:
    return (
        f"Closure of {d.ticket_id} was stopped ({d.action.upper()}): {d.reason}. "
        f"Reported: {ticket['complaint']} Work done: {ticket['action_taken']} Needs human inspection."
    )


def _execute_agent_with_model(
    model_instance: Model,
    ticket_id: str,
    guard: ClosureGuard,
    memo_source_label: str,
) -> None:
    """Run the Strands Agent with `guard` and tag any generated escalation memo with `memo_source_label`."""
    global _ACTIVE_MEMO_SOURCE
    _ACTIVE_MEMO_SOURCE = memo_source_label
    agent = Agent(
        model=model_instance,
        tools=[close_ticket, escalate_to_human],
        interventions=[guard],
        system_prompt=SYSTEM_PROMPT,
        callback_handler=None,
    )
    agent(f"Ticket {ticket_id} is eligible for auto-closure. Close it.")


def run_ticket(
    ticket_id: str,
    decider: Decider,
    policy: Policy = Policy(),
    offline: bool = False,
    log_path: str | None = None,
    provider: str = "auto",
    ticket_override: dict | None = None,
) -> dict:
    """Try to close one ticket.

    Modes (`provider`):
      - `offline=True` or `provider="offline"`: Guard only; no cloud LLM used; labeled template memo.
      - `provider="auto"` (default): Tries Amazon Bedrock first. If Bedrock is unavailable, throttled,
        or fails due to credentials/network AND `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) is configured,
        automatically fails over to the Google Gemini Strands Agent (with active model 503/timeout cascade).
      - `provider="bedrock"`: Strict Amazon Bedrock mode (no failover).
      - `provider="gemini"`: Runs the Strands Agent directly on Google Gemini API (with active model 503/timeout cascade).
    """
    ticket = ticket_override or TICKETS[ticket_id]
    lookup = (lambda tid: ticket if tid == ticket_id else TICKETS.get(tid)) if ticket_override else TICKETS.get
    STATE.pop(ticket_id, None)
    guard = ClosureGuard(decider, lookup, policy, log_path=log_path)

    effective_provider = "offline" if offline else (provider or "auto").lower()
    mode_used = effective_provider
    fallback_reason: Optional[str] = None
    continuity_fallback_used = False

    if effective_provider == "offline":
        d = decide(ticket, decider, policy)
        guard._record(d)
        if d.action == "allow":
            STATE[ticket_id] = {"status": "closed", "memo": "", "memo_source": ""}
        elif d.source != "rules":
            STATE[ticket_id] = {
                "status": "escalated",
                "memo": template_memo(ticket, d),
                "memo_source": "template (offline mode, Bedrock NOT used)",
            }
    elif effective_provider == "gemini":
        g_model = gemini_model()
        source_label = f"Google Gemini API ({g_model.model_id})"
        _execute_agent_with_model(g_model, ticket_id, guard, source_label)
        mode_used = "gemini"
        if g_model.used_continuity_fallback:
            continuity_fallback_used = True
            fallback_reason = g_model.continuity_error
    elif effective_provider in ("bedrock", "auto"):
        try:
            b_model = bedrock_model()
            _execute_agent_with_model(b_model, ticket_id, guard, "Amazon Bedrock model")
            mode_used = "bedrock"
        except Exception as bedrock_exc:
            if isinstance(bedrock_exc, BedrockUnavailable):
                primary_err = bedrock_exc
            elif isinstance(bedrock_exc, ModelThrottledException):
                primary_err = BedrockUnavailable(f"Bedrock throttled the request: {bedrock_exc}")
            else:
                primary_err = BedrockUnavailable(f"Bedrock request failed ({type(bedrock_exc).__name__}): {bedrock_exc}")

            # Automatic safety failover to Google Gemini API when in "auto" mode and a Gemini key is configured
            if effective_provider == "auto" and get_gemini_api_key():
                fallback_reason = str(primary_err)
                g_model = gemini_model()
                source_label = f"Google Gemini API ({g_model.model_id} safety fallback)"
                _execute_agent_with_model(g_model, ticket_id, guard, source_label)
                mode_used = "gemini_fallback"
                if g_model.used_continuity_fallback and g_model.continuity_error:
                    continuity_fallback_used = True
                    fallback_reason = f"{fallback_reason} | {g_model.continuity_error}"
            else:
                raise primary_err from bedrock_exc
    else:
        raise ValueError(f"Unknown provider mode: {provider}")

    return {
        "ticket_id": ticket_id,
        "mode": mode_used,
        "fallback_reason": fallback_reason,
        "continuity_fallback": continuity_fallback_used,
        "decision": guard.log[-1] if guard.log else None,
        "outcome": STATE.get(ticket_id, {"status": "open", "memo": "", "memo_source": ""}),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="*", help="ticket ids (default: all dev-split tickets)")
    ap.add_argument("--offline", action="store_true", help="guard only; neither Bedrock nor Gemini is used")
    ap.add_argument(
        "--provider",
        choices=["auto", "bedrock", "gemini", "offline"],
        default="auto",
        help="LLM provider for the closing agent & HITL memo (default: auto = Bedrock with Google Gemini safety fallback)",
    )
    ap.add_argument(
        "--gemini-decider-fallback",
        action="store_true",
        help="Also allow Google Gemini API to back up Layer-2 Decider 2B if port 8099 is unreachable",
    )
    args = ap.parse_args()
    ids = args.ids or [i for i, t in TICKETS.items() if t["split"] == "dev"]

    decider = Decider(gemini_fallback=True if args.gemini_decider_fallback else None)
    try:
        h = decider.health()
        chosen_mode = "offline" if args.offline else args.provider
        print(
            f"decider: {h.get('checkpoint')} [{h.get('device')}]  "
            f"agent-mode: {chosen_mode.upper()}  "
            f"gemini-safety-backup: {'READY' if get_gemini_api_key() else 'NOT SET'}\n"
        )
        for tid in ids:
            res = run_ticket(
                tid,
                decider,
                offline=args.offline,
                provider=args.provider,
                log_path=str(ROOT / "decisions.jsonl"),
            )
            d, out = res["decision"], res["outcome"]
            print(f"{tid}: {out['status']}  [mode={res['mode']}]")
            if res.get("fallback_reason"):
                print(f"   failover: {res['fallback_reason']}")
            if d:
                print(f"   guard={d.action.upper()} via {d.source}: {d.reason}")
                if d.scores:
                    s = d.scores
                    print(
                        f"   P(match)={s.p_match:.2f}  P(risk)={s.p_hazard:.2f}  "
                        f"decider={fmt_ms(s.decider_ms)} (round trip {fmt_ms(s.wall_ms)}) [{s.backend}]"
                    )
            if out["memo"]:
                print(f"   memo [{out['memo_source']}]: {out['memo']}")
            print()
    except (DeciderUnavailable, BedrockUnavailable, GeminiUnavailable) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())