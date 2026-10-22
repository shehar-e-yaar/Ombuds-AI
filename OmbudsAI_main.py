"""OmbudsAI: a reusable Strands Decider 2B guardrail for automated ticket closure,
with built-in adversarial prompt-injection defense and Google Gemini safety fallback.

THE DECISION PATTERN (copy this file into your own agent)
  What it does   Before an agent's `close_ticket` call runs, OmbudsAI checks deterministic
                 eligibility rules, screens the contractor log for prompt-injection gaming, and
                 asks Strands Decider 2B (with optional Google Gemini API safety backup) two
                 typed yes/no questions about the ground-truth ticket. Plain-Python policy turns
                 the two probabilities into allow / block / hold. The model classifies; code decides.
  Questions      QUESTIONS below (type `noul`: the answer is P(true), which is the confidence).
  Wiring         ClosureGuard, a Strands InterventionHandler: Agent(interventions=[guard]).
                 It returns Proceed (allow) or Guide (hand the turn back to the agent with the
                 facts, so the agent's own model — Amazon Bedrock or Google Gemini — writes the
                 reviewer memo).
  How to use     1. Run a decider server:  strands-decider serve <checkpoint> --port 8099
                    (or enable Google Gemini classifier backup via DELTASHIELD_GEMINI_DECIDER_FALLBACK=1)
                 2. Edit QUESTIONS, build_state() and Policy for your domain.
                 3. Fit the Policy thresholds on your own labeled dev set.
                 4. guard = ClosureGuard(Decider(), get_ticket=my_lookup)

EXPERIMENTAL. Thresholds are placeholders. Decider 2B is a small model: measure it yourself.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

try:
    from strands.interventions import Guide, InterventionHandler, Proceed
except ImportError:  # Lightweight compatibility shim when running unit tests without strands-agents installed

    class InterventionHandler:
        """Fallback InterventionHandler base class when strands-agents is not installed."""

    @dataclass
    class Proceed:
        reason: str = ""

    @dataclass
    class Guide:
        feedback: str = ""
        reason: str = ""


# ----------------------------------------------------------------------------- Google Gemini helpers (stdlib only)

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

# Ordered active v1beta model cascade (retired 1.5 / 2.0 endpoints removed so they never cause 404s)
GEMINI_MODEL_CASCADE = (
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.8-flash",
    "gemini-2.5-pro",
)

TRANSIENT_HTTP_CODES = {404, 408, 429, 500, 502, 503, 504}

_DISCOVERED_MODELS_CACHE: dict[str, list[str]] = {}
_LAST_WORKING_GEMINI_MODEL: Optional[str] = None

_NON_TEXT_MODEL_KEYWORDS = (
    "tts",
    "image",
    "embedding",
    "live",
    "veo",
    "imagen",
    "lyria",
    "robotics",
    "transcribe",
    "computer-use",
    "audio",
    "banana",
    "aqa",
    "learnlm",
)


class GeminiFallbackUnavailable(RuntimeError):
    """Google Gemini API key is missing or all Gemini fallback models refused the request."""


def get_gemini_api_key(explicit_key: Optional[str] = None) -> Optional[str]:
    """Return the configured Google Gemini API key from argument or environment."""
    key = explicit_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    return key.strip() if key and key.strip() else None


def get_gemini_model_id(explicit_model: Optional[str] = None) -> str:
    """Return the configured Google Gemini model ID (default: gemini-2.5-flash)."""
    model = explicit_model or os.environ.get("GEMINI_MODEL_ID") or DEFAULT_GEMINI_MODEL
    return model.strip()


def record_working_gemini_model(model_id: str) -> None:
    """Cache the most recently succeeded Gemini model so subsequent calls route to it first."""
    global _LAST_WORKING_GEMINI_MODEL
    if model_id and model_id.strip():
        _LAST_WORKING_GEMINI_MODEL = model_id.strip()


def iter_gemini_models(preferred: Optional[str] = None) -> list[str]:
    """Return ordered list of Gemini models starting with any known-working model and `preferred`,
    followed by the active fallback cascade."""
    primary = get_gemini_model_id(preferred)
    ordered: list[str] = []
    if _LAST_WORKING_GEMINI_MODEL and (preferred is None or preferred == DEFAULT_GEMINI_MODEL):
        ordered.append(_LAST_WORKING_GEMINI_MODEL)
    if primary not in ordered:
        ordered.append(primary)
    for candidate in GEMINI_MODEL_CASCADE:
        if candidate not in ordered:
            ordered.append(candidate)
    return ordered


def discover_available_gemini_models(api_key: str, timeout: float = 8.0) -> list[str]:
    """Query `GET /v1beta/models` (ModelService.ListModels) to discover active `generateContent`
    text models available to this API key."""
    if not api_key:
        return []
    cache_key = api_key[-8:]
    if cache_key in _DISCOVERED_MODELS_CACHE:
        return list(_DISCOVERED_MODELS_CACHE[cache_key])

    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        discovered: list[str] = []
        for m in payload.get("models") or []:
            methods = m.get("supportedGenerationMethods") or []
            if "generateContent" not in methods:
                continue
            raw_name = str(m.get("name") or "")
            short_name = raw_name.split("/", 1)[-1] if "/" in raw_name else raw_name
            lower_name = short_name.lower()
            if not lower_name.startswith("gemini-"):
                continue
            if any(kw in lower_name for kw in _NON_TEXT_MODEL_KEYWORDS):
                continue
            discovered.append(short_name)

        def _rank(name: str) -> tuple[int, str]:
            if name == "gemini-2.5-flash-lite":
                return (0, name)
            if name == "gemini-2.5-flash":
                return (1, name)
            if "flash" in name and "latest" in name:
                return (2, name)
            if "flash" in name:
                return (3, name)
            return (4, name)

        discovered.sort(key=_rank)
        _DISCOVERED_MODELS_CACHE[cache_key] = discovered
        return list(discovered)
    except Exception:
        return []


def gemini_generation_config(
    candidate_model: str,
    temperature: float = 0.0,
    max_output_tokens: int = 4096,
    response_mime_type: Optional[str] = None,
    include_thinking_config: bool = True,
) -> dict:
    """Build generationConfig for a Gemini model.

    Note: `gemini-2.5-flash` and `gemini-2.5-flash-lite` enable internal thinking tokens by default,
    which share `maxOutputTokens` and can truncate short outputs mid-sentence unless `thinkingBudget: 0`
    and a generous `maxOutputTokens` are set. (`gemini-2.5-pro` does not allow `thinkingBudget: 0`.)
    """
    cfg: dict = {
        "temperature": temperature,
        "maxOutputTokens": max(int(max_output_tokens), 2048),
    }
    if response_mime_type:
        cfg["responseMimeType"] = response_mime_type
    lower_model = candidate_model.lower()
    if include_thinking_config and ("2.5-flash" in lower_model):
        cfg["thinkingConfig"] = {"thinkingBudget": 0}
    return cfg


def post_gemini_generate_content(
    candidate_model: str,
    api_key: str,
    body: dict,
    timeout: float = 20.0,
) -> dict:
    """Send `POST /v1beta/models/{candidate_model}:generateContent` with automatic fallback:
      - If HTTP 400 occurs while `thinkingConfig` is present, strips `thinkingConfig` and retries immediately.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{candidate_model}:generateContent?key={api_key}"

    for attempt in range(2):
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # If a model rejects `thinkingConfig` with HTTP 400, strip it and retry once immediately
            gen_cfg = body.get("generationConfig") or {}
            if exc.code == 400 and "thinkingConfig" in gen_cfg and attempt == 0:
                new_cfg = dict(gen_cfg)
                new_cfg.pop("thinkingConfig", None)
                body = {**body, "generationConfig": new_cfg}
                continue
            raise


def ask_gemini_decider(
    state: str,
    questions: dict,
    api_key: Optional[str] = None,
    model_id: Optional[str] = None,
    timeout: float = 25.0,
) -> tuple[dict, float]:
    """Evaluate `noul` yes/no probability questions via Google Gemini REST API (stdlib only).

    Automatically cascades across `GEMINI_MODEL_CASCADE` (and `ListModels` discovery if needed)
    when a model returns HTTP 503 (high demand), HTTP 429 (rate limit), or a socket read timeout.
    """
    key = get_gemini_api_key(api_key)
    if not key:
        raise GeminiFallbackUnavailable(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set for Google Gemini safety fallback."
        )

    q_lines = []
    for q_name, q_spec in questions.items():
        crit = q_spec.get("criteria", {})
        q_lines.append(
            f'- "{q_name}": {q_spec.get("instructions", "")}\n'
            f'    * Return near 1.0 (true) if: {crit.get("true", "")}\n'
            f'    * Return near 0.0 (false) if: {crit.get("false", "")}'
        )

    prompt = (
        "You are a calibrated civic-safety probability classifier.\n"
        "Read the ticket state below and output a JSON object mapping each question key to a "
        "probability between 0.00 and 1.00 under the nested key \"noul\".\n"
        "Respond with valid JSON only, matching this exact structure:\n"
        "{\"work_addresses_report\": {\"noul\": 0.12}, \"report_is_safety_risk\": {\"noul\": 0.91}}\n\n"
        "QUESTIONS:\n" + "\n".join(q_lines) + "\n\n"
        "TICKET STATE:\n" + state
    )

    models_to_try = iter_gemini_models(model_id)
    errors_seen: list[str] = []

    idx = 0
    discovered_added = False
    while idx < len(models_to_try):
        candidate_model = models_to_try[idx]
        idx += 1
        per_model_timeout = min(timeout, 12.0) if idx == 1 else max(timeout, 25.0)
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": gemini_generation_config(
                candidate_model,
                temperature=0.0,
                max_output_tokens=2048,
                response_mime_type="application/json",
            ),
        }
        t0 = time.perf_counter()
        try:
            raw_payload = post_gemini_generate_content(
                candidate_model, key, body, timeout=per_model_timeout
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            candidates = raw_payload.get("candidates") or []
            parts = candidates[0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()
            parsed = json.loads(text)
            answers = {}
            for q_name in questions:
                val = parsed.get(q_name, {})
                prob = float(val.get("noul", val) if isinstance(val, dict) else val)
                answers[q_name] = {"noul": max(0.0, min(1.0, prob))}
            record_working_gemini_model(candidate_model)
            return answers, elapsed_ms
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
            err_msg = f"Google Gemini API ({candidate_model}) HTTP {exc.code}: {detail}"
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
            raise GeminiFallbackUnavailable(errors_seen[0]) from exc
        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            http.client.HTTPException,
            KeyError,
            IndexError,
            ValueError,
            TypeError,
        ) as exc:
            err_msg = f"Google Gemini API ({candidate_model}) {type(exc).__name__}: {exc}"
            errors_seen.append(err_msg)
            if idx == len(models_to_try) and not discovered_added:
                discovered_added = True
                for dm in discover_available_gemini_models(key):
                    if dm not in models_to_try:
                        models_to_try.append(dm)
            if idx < len(models_to_try):
                time.sleep(0.15)
                continue
            raise GeminiFallbackUnavailable(errors_seen[0]) from exc

    raise GeminiFallbackUnavailable(errors_seen[0] if errors_seen else "All Google Gemini fallback models failed.")


# ----------------------------------------------------------------------------- decider client


class DeciderUnavailable(RuntimeError):
    """The decider server is not answering."""


class Decider:
    """Tiny client for a running `strands-decider serve` (POST /v1/systemone). Stdlib only.

    Modeled on examples/strands/_client.py in strands-labs/strands-decider (Apache-2.0).
    Supports optional automatic fallback to Google Gemini API when `gemini_fallback=True`
    or `DELTASHIELD_GEMINI_DECIDER_FALLBACK=1` is set in the environment.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        timeout: float = 300.0,
        gemini_fallback: Optional[bool] = None,
        gemini_api_key: Optional[str] = None,
        gemini_model: Optional[str] = None,
    ) -> None:
        self.url = (url or os.environ.get("STRANDS_DECIDER_URL", "http://127.0.0.1:8099")).rstrip("/")
        self.timeout = timeout
        if gemini_fallback is None:
            env_flag = os.environ.get("DELTASHIELD_GEMINI_DECIDER_FALLBACK", "").strip().lower()
            self.gemini_fallback = env_flag in ("1", "true", "yes", "on")
        else:
            self.gemini_fallback = bool(gemini_fallback)
        self.gemini_api_key = gemini_api_key
        self.gemini_model = gemini_model
        self.last_ms: Optional[float] = None  # server-side inference time of the last call
        self.last_backend: str = "strands-decider-2b"

    def _call(self, path: str, body: Optional[dict] = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            self.url + path, data=data, headers={"content-type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
            raise DeciderUnavailable(
                f"no decider server at {self.url} ({exc}). Start one with:\n"
                "    strands-decider serve StrandsAgents/strands-decider-2B-hobson-v21 --port 8099"
            ) from exc

    def health(self) -> dict:
        try:
            info = self._call("/health")
            self.last_backend = "strands-decider-2b"
            return info
        except DeciderUnavailable:
            if self.gemini_fallback and get_gemini_api_key(self.gemini_api_key):
                model = get_gemini_model_id(self.gemini_model)
                self.last_backend = f"gemini-safety-fallback ({model})"
                return {
                    "status": "fallback",
                    "checkpoint": f"google-gemini/{model}",
                    "device": "Google Gemini Cloud Safety Backup",
                    "fallback": True,
                }
            raise

    def ask(self, state: str, questions: dict) -> dict:
        try:
            payload = self._call("/v1/systemone", {"state": state, "model": "hobson-latest", "questions": questions})
            self.last_ms = payload.get("latency_ms")
            self.last_backend = "strands-decider-2b"
            return payload["answers"]
        except DeciderUnavailable as primary_exc:
            if self.gemini_fallback and get_gemini_api_key(self.gemini_api_key):
                try:
                    answers, elapsed_ms = ask_gemini_decider(
                        state,
                        questions,
                        api_key=self.gemini_api_key,
                        model_id=self.gemini_model,
                        timeout=min(self.timeout, 30.0),
                    )
                    self.last_ms = elapsed_ms
                    self.last_backend = f"gemini-safety-fallback ({get_gemini_model_id(self.gemini_model)})"
                    return answers
                except GeminiFallbackUnavailable as gem_exc:
                    raise DeciderUnavailable(f"{primary_exc} | Gemini safety backup also failed: {gem_exc}") from gem_exc
            raise


class GeminiDecider(Decider):
    """Direct Google Gemini classifier implementing the exact Decider interface."""

    def __init__(self, api_key: Optional[str] = None, model_id: Optional[str] = None, timeout: float = 30.0) -> None:
        super().__init__(timeout=timeout, gemini_fallback=True, gemini_api_key=api_key, gemini_model=model_id)

    def health(self) -> dict:
        key = get_gemini_api_key(self.gemini_api_key)
        if not key:
            raise DeciderUnavailable("GEMINI_API_KEY is not set for GeminiDecider.")
        model = get_gemini_model_id(self.gemini_model)
        self.last_backend = f"google-gemini ({model})"
        return {"status": "ok", "checkpoint": f"google-gemini/{model}", "device": "Google Gemini API"}

    def ask(self, state: str, questions: dict) -> dict:
        try:
            answers, elapsed_ms = ask_gemini_decider(
                state, questions, api_key=self.gemini_api_key, model_id=self.gemini_model, timeout=self.timeout
            )
            self.last_ms = elapsed_ms
            self.last_backend = f"google-gemini ({get_gemini_model_id(self.gemini_model)})"
            return answers
        except GeminiFallbackUnavailable as exc:
            raise DeciderUnavailable(str(exc)) from exc


def fmt_ms(ms: Optional[float]) -> str:
    """142 -> '142 ms', 61081 -> '61.1 s'."""
    if ms is None:
        return "n/a"
    return f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000:.1f} s"


def yes_no(instructions: str, if_true: str, if_false: str) -> dict:
    return {"type": "noul", "instructions": instructions, "criteria": {"true": if_true, "false": if_false}}


# ----------------------------------------------------------------------------- 1. the questions

QUESTIONS = {
    "work_addresses_report": yes_no(
        "Does the work the contractor carried out address the problem the citizen reported?",
        "the work targets the root cause of the reported problem",
        "the work is about something else, or treats only a side symptom, "
        "so the reported problem may still be there",
    ),
    "report_is_safety_risk": yes_no(
        "Does the citizen's report describe a risk of physical harm to people or animals?",
        "the report describes danger such as electric shock, fire, gas, flooding, "
        "an open hole, a collapse or an injury risk",
        "the report describes a nuisance or inconvenience with no physical danger",
    ),
}

# ----------------------------------------------------------------------------- layer 1: rules & anti-injection floor

MIN_HOURS = 48.0

# Patterns that indicate a contractor is attempting prompt injection or rubric gaming in the work log
_ADVERSARIAL_PATTERNS = [
    (re.compile(r"ignore\s+(?:all\s+|previous\s+|prior\s+)?(?:instructions|rules|prompts|citizen|report)", re.I),
     "instruction-override attempt"),
    (re.compile(r"system\s*override|\[system\]|<system>|developer\s*mode", re.I),
     "spoofed system directive"),
    (re.compile(r"\bp_match\b|\bp_hazard\b|\bshould_auto_close\b|\bwork_addresses_report\b", re.I),
     "internal scoring variable manipulation"),
    (re.compile(r"do\s+not\s+escalate|force\s+allow|auto[\s_-]*approve\s+immediately", re.I),
     "guardrail bypass command"),
]


def rules_check(ticket: dict) -> tuple[bool, str]:
    """Deterministic floor: the checks a plain auto-closer already does."""
    if ticket.get("status") != "COMPLETED":
        return False, f"work status is {ticket.get('status')}, not COMPLETED"
    hours = float(ticket.get("hours_elapsed", 0))
    if hours < MIN_HOURS:
        return False, f"only {hours:.1f}h since completion (minimum {MIN_HOURS:.0f}h)"
    if not (ticket.get("action_taken") or "").strip():
        return False, "no work description recorded"
    return True, f"completed, {hours:.1f}h since completion (minimum {MIN_HOURS:.0f}h)"


def adversarial_check(ticket: dict) -> tuple[bool, str]:
    """Screen contractor work logs for adversarial prompt injection or rubric gaming."""
    action_text = str(ticket.get("action_taken") or "")
    for pattern, label in _ADVERSARIAL_PATTERNS:
        if pattern.search(action_text):
            return False, f"adversarial contractor log detected ({label})"
    return True, "no adversarial patterns detected"


def build_state(ticket: dict, rules_reason: str) -> str:
    """The text Decider reads. The rule verdict is an INPUT to the model, not only a gate."""
    parts = [
        "A citizen reported a problem to a city department. A contractor says the work is done. "
        "An automated agent now wants to close the ticket.",
        "",
        "--- CITIZEN REPORT ---",
        ticket["complaint"],
    ]
    if ticket.get("citizen_followup"):
        parts += ["", "--- LATEST CITIZEN MESSAGE ---", ticket["citizen_followup"]]
    parts += [
        "",
        "--- WORK DONE (contractor log) ---",
        ticket["action_taken"],
        "",
        "--- AUTOMATED RULE CHECK ---",
        f"passed: {rules_reason}",
    ]
    return "\n".join(parts)


# ----------------------------------------------------------------------------- 2. the policy


@dataclass(frozen=True)
class Policy:
    """Placeholders. Fit on the dev split, then run the test split once."""

    hazard_yes: float = 0.50          # P(safety risk) at/above this => safety-critical report
    match_allow: float = 0.65         # P(work addresses report) needed to allow a routine ticket
    match_allow_hazard: float = 0.85  # ...needed to allow a safety-critical ticket
    mismatch_block: float = 0.25      # at/below this => clear mismatch => block


def apply_policy(p_match: float, p_hazard: float, policy: Policy) -> tuple[str, str]:
    """Probabilities -> 'allow' | 'block' | 'hold'. Low confidence means unsure, not dangerous."""
    needed = policy.match_allow_hazard if p_hazard >= policy.hazard_yes else policy.match_allow
    if p_match >= needed:
        return "allow", f"work matches report (P={p_match:.2f} >= {needed:.2f})"
    if p_match <= policy.mismatch_block:
        return "block", f"work does not match report (P={p_match:.2f} <= {policy.mismatch_block:.2f})"
    return "hold", f"evidence unclear (P={p_match:.2f}, needed {needed:.2f} to allow)"


def policy_threshold_for(p_hazard: float, policy: Policy = Policy()) -> tuple[float, str]:
    """Return `(required_match_threshold, risk_tier_label)` for a given hazard probability."""
    if p_hazard >= policy.hazard_yes:
        return policy.match_allow_hazard, "SAFETY-CRITICAL"
    return policy.match_allow, "ROUTINE"


@dataclass
class Scores:
    p_match: float
    p_hazard: float
    decider_ms: Optional[float]  # server-side inference time
    wall_ms: float               # round trip including HTTP
    backend: str = "strands-decider-2b"


@dataclass
class Decision:
    ticket_id: str
    action: str   # allow | block | hold
    source: str   # rules | security | decider
    reason: str
    scores: Optional[Scores] = None


def score_ticket(ticket: dict, decider: Decider, rules_reason: str) -> Scores:
    start = time.perf_counter()
    answers = decider.ask(build_state(ticket, rules_reason), QUESTIONS)
    wall = (time.perf_counter() - start) * 1000
    backend = getattr(decider, "last_backend", "strands-decider-2b")
    return Scores(
        answers["work_addresses_report"]["noul"],
        answers["report_is_safety_risk"]["noul"],
        decider.last_ms,
        wall,
        backend=backend,
    )


def decide(ticket: dict, decider: Decider, policy: Policy = Policy()) -> Decision:
    ok, why = rules_check(ticket)
    if not ok:
        return Decision(ticket["ticket_id"], "hold", "rules", f"rule check failed: {why}")
    clean, sec_why = adversarial_check(ticket)
    if not clean:
        return Decision(ticket["ticket_id"], "block", "security", f"security check failed: {sec_why}")
    scores = score_ticket(ticket, decider, why)
    action, reason = apply_policy(scores.p_match, scores.p_hazard, policy)
    return Decision(ticket["ticket_id"], action, "decider", reason, scores)


# ----------------------------------------------------------------------------- 3. the handler


class ClosureGuard(InterventionHandler):
    """Runs before every tool call; only `guarded_tool` is reviewed.

    The ticket is looked up by id in YOUR system of record, so Decider reads the real
    complaint and work log, not text the agent may have paraphrased.
    """

    name = "deltashield-closure-guard"

    def __init__(
        self,
        decider: Decider,
        get_ticket: Callable[[str], Optional[dict]],
        policy: Policy = Policy(),
        guarded_tool: str = "close_ticket",
        id_arg: str = "ticket_id",
        escalation_tool: str = "escalate_to_human",
        log_path: Optional[str] = None,
    ) -> None:
        self.decider, self.get_ticket, self.policy = decider, get_ticket, policy
        self.guarded_tool, self.id_arg, self.escalation_tool = guarded_tool, id_arg, escalation_tool
        self.log_path = Path(log_path) if log_path else None
        self.log: list[Decision] = []
        self._cache: dict[str, Decision] = {}

    def before_tool_call(self, event, **kwargs):
        if event.tool_use["name"] != self.guarded_tool:
            return Proceed()
        tid = str((event.tool_use.get("input") or {}).get(self.id_arg, ""))
        ticket = self.get_ticket(tid)
        if ticket is None:
            return Guide(feedback=f"Ticket '{tid}' does not exist. Do not close anything.")
        if tid in self._cache:
            decision = self._cache[tid]
        else:
            decision = decide(ticket, self.decider, self.policy)
            self._cache[tid] = decision
            self._record(decision)
        if decision.action == "allow":
            return Proceed(reason=decision.reason)
        return Guide(feedback=self.feedback(ticket, decision), reason=decision.reason)

    def feedback(self, ticket: dict, d: Decision) -> str:
        if d.source == "rules":
            return f"Ticket {d.ticket_id} is not eligible for closure: {d.reason}. Do not close it or retry."
        if d.source == "security":
            verdict = "the contractor log contains a suspected prompt-injection or rubric-gaming attempt"
        elif d.action == "block":
            verdict = "the recorded work does not appear to address what the citizen reported"
        else:
            verdict = "it is unclear whether the recorded work addresses what the citizen reported"
        return (
            f"Closure of {d.ticket_id} was stopped by the safety guard: {verdict}. "
            f"Do not close it or retry. Call {self.escalation_tool} with the ticket_id and a "
            "memo of 2-3 sentences for a human reviewer. Use only these facts:\n"
            f"  Citizen reported: {ticket['complaint']}\n"
            f"  Contractor log: {ticket['action_taken']}\n"
            f"  Guard result: {d.action.upper()} - {d.reason}"
        )

    def _record(self, d: Decision) -> None:
        self.log.append(d)
        if self.log_path:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(d)) + "\n")