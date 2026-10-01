"""Gemini text generation with model fallback.

Why this exists
---------------
Every Gemini call in the app used to hardcode `gemini-2.5-flash`. On a real
account that model can be over its quota (HTTP 429) while the current models
answer fine — measured on this machine, `gemini-2.5-flash` returns
`RESOURCE_EXHAUSTED` on every request while `gemini-3.8-flash` and
`gemini-3.5-flash-lite` return 200 immediately. Pinning one slug therefore
turns a working Gemini connection into a completely dead assistant.

So this walks an ordered list and quarantines the models that cannot answer:

* 404 NOT_FOUND — the slug is retired. Quarantine for `RETIRED_TTL`.
* 429 RESOURCE_EXHAUSTED / quota — the account is out of allowance for that
  model. Quarantine for `QUOTA_TTL`, which is short, because it is per-model
  and can differ between slugs on the same key.
* 503 UNAVAILABLE — capacity spike, transient. Retry it sooner (`BUSY_TTL`).

Successful models are remembered, so the steady state costs one request.
"""

import logging
import threading
import time

logger = logging.getLogger("gemini_client")

# Ordered best-first. Cheap/flash models lead because this backs an interactive
# assistant; stronger models stay in the list as quota on the first ones runs out.
GEMINI_MODELS: list[str] = [
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-2.5-flash",
]

RETIRED_TTL = 6 * 60 * 60   # a 404 slug stays gone
QUOTA_TTL = 15 * 60        # per-model allowance; recover sooner than a full day
BUSY_TTL = 60              # capacity spikes are usually brief


class GeminiUnavailable(RuntimeError):
    """No Gemini model could answer right now.

    `note` is a short, user-facing explanation of what actually happened, which
    is what the caller needs in order to tell the user what to do about it.
    """


_quarantined: dict[str, float] = {}
_quarantine_lock = threading.RLock()

# The model that most recently answered, so the next call can try it first.
_last_good: str | None = None


def _remember(model: str) -> None:
    global _last_good
    with _quarantine_lock:
        _last_good = model


def last_good_model() -> str | None:
    return _last_good


def _is_quarantined(model: str) -> bool:
    with _quarantine_lock:
        until = _quarantined.get(model)
        if until is None:
            return False
        if time.time() >= until:
            del _quarantined[model]
            return False
        return True


def _quarantine(model: str, seconds: float, reason: str) -> None:
    with _quarantine_lock:
        _quarantined[model] = time.time() + seconds
    logger.info(f"[Gemini] Quarantined {model} for {int(seconds)}s ({reason})")


def quarantined_models() -> dict[str, float]:
    """Model -> seconds left, for diagnostics."""
    now = time.time()
    with _quarantine_lock:
        return {m: max(0, int(t - now)) for m, t in _quarantined.items() if t > now}


def reset_quarantine() -> None:
    global _last_good
    with _quarantine_lock:
        _quarantined.clear()
        _last_good = None


def extract_text(response) -> str:
    """Pull the text out of a generate_content response.

    Prefers the structured candidate walk because `response.text` collapses
    multiple parts and can come back empty for some shapes.
    """
    parts_text: list[str] = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []) or []:
            text = getattr(part, "text", None)
            if text:
                parts_text.append(text)

    text = "".join(parts_text).strip()
    if text:
        return text
    try:
        return (getattr(response, "text", "") or "").strip()
    except Exception:
        return ""


def _classify(exc: Exception) -> tuple[str, str]:
    """Return (kind, note) where kind is 'retired' | 'quota' | 'busy' | 'error'."""
    text = str(exc)
    low = text.lower()

    if "404" in low or "not_found" in low or "no longer available" in low:
        return "retired", "model retired"
    if "429" in low or "resource_exhausted" in low or "quota" in low or "rate limit" in low:
        return "quota", "out of quota"
    if "503" in low or "unavailable" in low or "high demand" in low or "overloaded" in low:
        return "busy", "capacity spike"
    if "401" in low or "unauthenticated" in low or "api_key_invalid" in low or \
            "access_token_type_unsupported" in low or "permission denied" in low:
        # Credential-level, not model-level: quarantining the slug would not help.
        return "error", "authentication failed"
    return "error", "request failed"


def generate(
    prompt: str,
    system: str = "",
    temperature: float = 0.6,
    max_output_tokens: int | None = None,
    models: list[str] | None = None,
    api_key: str | None = None,
    max_models: int | None = None,
) -> str:
    """Generate text, walking models until one answers.

    `max_models` bounds the walk. The health probe passes a small value so a
    rate-limited account cannot turn a UI button into an 80-second hang; the
    assistant itself leaves it unset and will try the whole list.

    Raises `GeminiUnavailable` when every candidate model is exhausted, with a
    note that distinguishes the cases a user can act on: the credential is bad,
    the account is out of allowance, or the models are merely busy.
    """
    if api_key is None:
        from main import _get_api_key
        api_key = _get_api_key()

    if not api_key:
        raise GeminiUnavailable(
            "No Gemini API key configured. Get a free key at "
            "https://aistudio.google.com/apikey"
        )

    from google import genai

    # The SDK retries 429/503 internally with exponential backoff, which costs
    # ~30s per dead model before this code ever gets to try the next one. We
    # already have a better strategy — quarantine the slug and move on — so a
    # single attempt per model keeps failover fast.
    http_options: dict = {"api_version": "v1beta"}
    try:
        from google.genai import types as genai_types
        http_options["retry_options"] = genai_types.HttpRetryOptions(
            attempts=1,
            http_status_codes=[429, 500, 502, 503, 504],
        )
    except Exception:
        # Older/newer SDK without retry options: single attempts are still
        # handled correctly, just more slowly on transient errors.
        pass

    client = genai.Client(api_key=api_key, http_options=http_options)

    config: dict = {"temperature": temperature}
    if max_output_tokens:
        config["max_output_tokens"] = max_output_tokens

    contents = f"{system}\n\n{prompt}" if system else prompt

    candidates = list(models or GEMINI_MODELS)

    # Try the model that last worked first: in steady state that makes the walk
    # exactly one request instead of re-probing the head of the list every time.
    if _last_good in candidates:
        candidates.remove(_last_good)
        candidates.insert(0, _last_good)

    ordered = [m for m in candidates if not _is_quarantined(m)]
    if not ordered:
        reset_quarantine()
        ordered = list(candidates)

    if max_models:
        ordered = ordered[:max_models]

    last_error: Exception | None = None
    kinds: list[str] = []

    for model in ordered:
        try:
            response = client.models.generate_content(
                model=model, contents=contents, config=config,
            )
            text = extract_text(response)
            if text:
                _remember(model)
                return text
            # A 200 with empty text is not a hard failure: it usually means the
            # token budget was too small to emit anything. Try another model,
            # but only briefly, because the next model may well behave the same.
            last_error = RuntimeError(f"{model} returned an empty response")
            kinds.append("empty")
            _quarantine(model, BUSY_TTL, "empty response")
        except Exception as e:
            last_error = e
            kind, reason = _classify(e)
            kinds.append(kind)
            if kind == "retired":
                _quarantine(model, RETIRED_TTL, reason)
            elif kind == "quota":
                _quarantine(model, QUOTA_TTL, reason)
            elif kind == "busy":
                _quarantine(model, BUSY_TTL, reason)
            else:
                # Credential-level failure: no other slug will fare better, and
                # retrying them all just multiplies the latency.
                break

    if kinds and set(kinds) == {"quota"}:
        raise GeminiUnavailable(
            "Gemini accepted the key but the account has no allowance left for any "
            "of its models. Wait for the quota to reset, or enable billing in "
            "Google AI Studio."
        ) from last_error

    if kinds and set(kinds) == {"empty"}:
        raise GeminiUnavailable(
            "Gemini answered but returned no text for every model. This usually "
            "means the models are returning empty content right now — try again "
            "shortly."
        ) from last_error

    detail = str(last_error)[:200] if last_error else "no model available"
    raise GeminiUnavailable(f"No Gemini model could answer. {detail}") from last_error