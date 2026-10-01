"""Single source of truth for "can this app actually answer right now?".

Every other module answers this question badly and differently: `or_client`
walks a model pool, `llm_client` branches on a settings string, `main.py`
hand-rolls a Gemini-then-OpenRouter chain per call site, and the Settings UI's
"Test Connection" button only checks that a key string is non-empty. When all of
them fail at once the user is left guessing which of the three things to fix.

This module probes each provider directly and answers with a *state*, a *detail*
line, and a *fix* line, so every caller and the UI can report the same truth.

States
------
ok             configured, reachable, and answered a real request
missing        no key / nothing configured
invalid_key    key present but the wrong shape or rejected (401/403)
quota          account-level daily allowance used up (OpenRouter free tier)
rate_limited   transient per-model load; a retry later may succeed
offline        host unreachable / connection error
error          anything else

Probe cost
----------
A "deep" probe sends one real generation request with `max_output_tokens=1` /
`max_tokens=1`. That is the only way to tell "key is fine" apart from "key is
fine but the account has no quota left", which is precisely the state that
matters most. Deep results are cached for `DEEP_TTL` so the cost is at most one
near-worthless token per provider every ten minutes.
"""

import json
import logging
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import requests

from core.user_paths import get_user_data_dir

logger = logging.getLogger("provider_health")

API_KEY_PATH = get_user_data_dir() / "config" / "api_keys.json"
SETTINGS_PATH = get_user_data_dir() / "config" / "app_settings.json"

# ── States ───────────────────────────────────────────────────────────────────
STATE_OK = "ok"
STATE_MISSING = "missing"
STATE_INVALID_KEY = "invalid_key"
STATE_QUOTA = "quota"
STATE_RATE_LIMITED = "rate_limited"
STATE_OFFLINE = "offline"
STATE_ERROR = "error"

# States in which a provider is expected to answer a real request.
USABLE_STATES = {STATE_OK, STATE_RATE_LIMITED}

GEMINI_PROBE_MODEL = "gemini-3.8-flash"

SHALLOW_TTL = 90.0    # key-shape / reachability only
DEEP_TTL = 600.0     # includes one real generation request


@dataclass
class ProviderStatus:
    """The verdict for one provider."""

    name: str
    state: str = STATE_MISSING
    detail: str = "Not checked yet."
    fix: str = ""
    latency_ms: Optional[int] = None
    checked_at: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return self.state in USABLE_STATES

    @property
    def checked(self) -> bool:
        return self.checked_at > 0.0

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "state": self.state,
            "detail": self.detail,
            "fix": self.fix,
            "latency_ms": self.latency_ms,
            "checked_at": self.checked_at,
            "extra": dict(self.extra),
            "usable": self.usable,
        }

    def one_line(self) -> str:
        mark = "OK " if self.state == STATE_OK else "!! "
        lat = f" ({self.latency_ms} ms)" if self.latency_ms else ""
        return f"{mark}{self.name}: {self.detail}{lat}"


# ── Config loading ───────────────────────────────────────────────────────────

def _load_keys() -> dict:
    try:
        with open(API_KEY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.warning(f"[Health] Could not read api_keys.json: {e}")
        return {}


def _load_settings() -> dict:
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def gemini_key() -> str:
    return (_load_keys().get("gemini_api_key") or "").strip()


def openrouter_key() -> str:
    return (_load_keys().get("openrouter_api_key") or "").strip()


def gemini_key_shape_ok(key: str) -> bool:
    """True when the credential looks like a classic AI Studio key ('AIza…').

    This is a *hint*, not a verdict. Google also accepts project-bound
    OAuth/access tokens, which work perfectly once the Generative Language API
    is enabled for their project, so callers must never use this to skip a
    credential — only to describe it.
    """
    return bool(key) and key.startswith("AIza")


def openrouter_key_shape_ok(key: str) -> bool:
    return key.startswith("sk-or-")


# ── Classification helpers ───────────────────────────────────────────────────

def _is_connection_error(e: Exception) -> bool:
    return isinstance(e, requests.exceptions.ConnectionError) or \
        isinstance(e, socket.timeout) or isinstance(e, TimeoutError)


def _is_daily_quota_text(text: str) -> bool:
    low = (text or "").lower()
    return "free-models-per-day" in low or \
        ("daily limit" in low and "free" in low) or \
        "insufficient_quota" in low


def _classify_auth(text: str) -> bool:
    low = (text or "").lower()
    return "unauthenticated" in low or "api_key_invalid" in low or \
        "invalid api key" in low or "access_token_type_unsupported" in low or \
        "permission denied" in low


# ── Probes ───────────────────────────────────────────────────────────────────

def probe_gemini(deep: bool = True) -> ProviderStatus:
    """Check the Gemini credential and, when `deep`, send one 1-token generation.

    The credential shape is deliberately *not* used to reject a key. Google
    accepts AI Studio keys (`AIza…`) as well as project-bound OAuth/access
    tokens, and the latter really do answer requests once the Generative
    Language API is enabled for their project. An earlier version of this file
    short-circuited on a missing `AIza` prefix and so reported a working
    credential as unusable — permanently ignoring a key that works.

    The only authority on whether a credential is good is Google's response, so
    that is what we ask.
    """
    key = gemini_key()
    status = ProviderStatus("Google Gemini")

    if not key:
        status.state = STATE_MISSING
        status.detail = "No Gemini API key configured."
        status.fix = "Get a free key at https://aistudio.google.com/apikey"
        return status

    unusual = not gemini_key_shape_ok(key)
    if unusual:
        status.extra["unusual_credential"] = True
        status.detail = "Testing a non-AIza credential (OAuth/access token)..."
    else:
        status.detail = "Key format looks valid."

    if not deep:
        status.state = STATE_OK
        return status

    started = time.time()
    # Imported before the try so the except branch can reference it even if the
    # import itself is what failed.
    import gemini_client

    try:
        # Use the same multi-model client the app itself uses. Probing a single
        # slug would report "quota exhausted" on an account whose other models
        # answer fine, which is precisely the state that made this look broken.
        gemini_client.generate(
            "Reply with the single word: ok",
            # Deliberately no max_output_tokens cap. Gemini 3.x models emit
            # "thinking" tokens before the answer, so a small output budget is
            # consumed by reasoning and the call returns HTTP 200 with an empty
            # string — which reads as "no answer" and makes a healthy account
            # look out of quota. A short uncapped "ping" costs a handful of tokens.
            temperature=0.0,
            api_key=key,
            # Keep a UI button responsive: on a rate-limited account the full
            # walk can take over a minute.
            max_models=3,
        )
    except Exception as e:
        status.latency_ms = int((time.time() - started) * 1000)
        text = str(e)
        if isinstance(e, gemini_client.GeminiUnavailable) and "allowance" in str(e):
            status.state = STATE_QUOTA
            status.detail = str(e)
            status.fix = (
                "Wait for the quota to reset, or enable billing in Google AI Studio "
                "(https://aistudio.google.com)."
            )
            return status
        if _is_connection_error(e):
            status.state = STATE_OFFLINE
            status.detail = "Could not reach Google's API."
            status.fix = "Check your internet connection."
        elif _is_daily_quota_text(text) or "429" in text or "RESOURCE_EXHAUSTED" in text:
            status.state = STATE_QUOTA
            status.detail = "Gemini rejected the request: rate limit or quota exhausted."
            status.fix = "Wait for the quota to reset, or add billing in Google AI Studio."
        elif _classify_auth(text):
            status.state = STATE_INVALID_KEY
            status.detail = "Google rejected this key (401/403)."
            status.fix = "Check the key at https://aistudio.google.com/apikey"
        else:
            status.state = STATE_ERROR
            status.detail = f"Gemini request failed: {text[:200]}"
        return status

    status.latency_ms = int((time.time() - started) * 1000)
    status.state = STATE_OK
    if unusual:
        # Worth surfacing: this works, but it is not a standalone AI Studio key,
        # so it cannot be relied on across machines or re-auths.
        status.detail = (
            "Responded to a live request (using an OAuth/access token rather "
            "than an AI Studio key)."
        )
    else:
        status.detail = "Responded to a live request."
    return status


def probe_openrouter(deep: bool = True) -> ProviderStatus:
    """Validate the key, then (deep) confirm a free model can still answer.

    The key check alone is not enough: an account can have a perfectly valid key
    and zero free-model allowance left for the day. Only a real completion
    reveals that, and it is the failure mode that wastes the most user time
    because the key *looks* fine in Settings.
    """
    key = openrouter_key()
    status = ProviderStatus("OpenRouter")

    if not key:
        status.state = STATE_MISSING
        status.detail = "No OpenRouter API key configured."
        status.fix = "Create a key at https://openrouter.ai/keys"
        return status

    if not openrouter_key_shape_ok(key):
        status.state = STATE_INVALID_KEY
        status.detail = (
            "The stored key does not start with 'sk-or-', so OpenRouter will "
            "reject it."
        )
        status.fix = "Replace it with a key from https://openrouter.ai/keys"
        return status

    # Cheap, non-billable reachability + key check.
    started = time.time()
    try:
        resp = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=15,
        )
    except Exception as e:
        status.latency_ms = int((time.time() - started) * 1000)
        if _is_connection_error(e):
            status.state = STATE_OFFLINE
            status.detail = "Could not reach openrouter.ai."
            status.fix = "Check your internet connection."
        else:
            status.state = STATE_ERROR
            status.detail = f"OpenRouter request failed: {str(e)[:200]}"
        return status

    status.latency_ms = int((time.time() - started) * 1000)

    if resp.status_code == 401:
        status.state = STATE_INVALID_KEY
        status.detail = "OpenRouter rejected this key (HTTP 401)."
        status.fix = "Replace it with a key from https://openrouter.ai/keys"
        return status
    if resp.status_code >= 400:
        status.state = STATE_ERROR
        status.detail = f"OpenRouter returned HTTP {resp.status_code}."
        return status

    try:
        free_models = [
            m["id"] for m in resp.json().get("data", [])
            if ":free" in m.get("id", "")
        ]
    except Exception:
        free_models = []
    status.extra["free_model_count"] = len(free_models)

    if not free_models:
        status.state = STATE_ERROR
        status.detail = "OpenRouter is not listing any free models right now."
        return status

    if not deep:
        status.state = STATE_OK
        status.detail = f"Key valid; {len(free_models)} free models listed."
        return status

    # Deep: one 1-token completion against the first advertised free model.
    started = time.time()
    try:
        comp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/brahma-ai",
                "X-Title": "Brahma Evo",
            },
            json={
                "model": free_models[0],
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
            },
            timeout=30,
        )
    except Exception as e:
        status.latency_ms = int((time.time() - started) * 1000)
        if _is_connection_error(e):
            status.state = STATE_OFFLINE
            status.detail = "Could not reach openrouter.ai."
            status.fix = "Check your internet connection."
        else:
            status.state = STATE_ERROR
            status.detail = f"OpenRouter request failed: {str(e)[:200]}"
        return status

    status.latency_ms = int((time.time() - started) * 1000)

    if comp.status_code == 200:
        status.state = STATE_OK
        status.detail = f"Responded to a live request ({free_models[0]})."
        return status

    body = comp.text or ""
    if _is_daily_quota_text(body):
        status.state = STATE_QUOTA
        status.detail = (
            "The account's daily free-model allowance is used up "
            "(free-models-per-day). The key is valid — every free model is "
            "returning 429 until the daily reset."
        )
        status.fix = (
            "Add credits at https://openrouter.ai/credits, or use a free "
            "Gemini AI Studio key (https://aistudio.google.com/apikey), or "
            "wait for the daily reset."
        )
        return status

    if comp.status_code == 429:
        status.state = STATE_RATE_LIMITED
        status.detail = "Rate limited right now (429). A retry may succeed."
        return status

    if comp.status_code in (401, 403):
        status.state = STATE_INVALID_KEY
        status.detail = f"OpenRouter rejected this key (HTTP {comp.status_code})."
        status.fix = "Check the key and account at https://openrouter.ai/keys"
        return status

    if comp.status_code == 404:
        # The advertised model was retired between the listing and the call.
        # That is not a key problem, so do not report it as one.
        status.state = STATE_RATE_LIMITED
        status.detail = "The first free model was retired; retry will pick another."
        return status

    status.state = STATE_ERROR
    status.detail = f"OpenRouter returned HTTP {comp.status_code}: {body[:150]}"
    return status


def _tcp_open(host: str, port: int, timeout: float = 0.4) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            return s.connect_ex((host, port)) == 0
    except Exception:
        return False


def probe_local(deep: bool = True) -> ProviderStatus:
    """Detect a local OpenAI-compatible server (Ollama / LM Studio / llama.cpp)."""
    settings = _load_settings()
    url = (settings.get("local_ai_url") or "http://localhost:11434/v1").rstrip("/")
    model = settings.get("local_ai_model") or "llama3.2"
    status = ProviderStatus("Local (Ollama / LM Studio)")
    status.extra["url"] = url
    status.extra["configured_model"] = model

    # Parse host/port out of the URL for a cheap socket probe first.
    host, port = "localhost", 11434
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except Exception:
        pass

    if not _tcp_open(host, port):
        status.state = STATE_OFFLINE
        status.detail = f"No local server is listening on {url}."
        status.fix = (
            "Start Ollama (ollama serve, then 'ollama pull llama3.2') or "
            "LM Studio, then set the URL and model in Settings -> Local AI."
        )
        return status

    if not deep:
        status.state = STATE_OK
        status.detail = f"A local server is listening on {url}."
        return status

    started = time.time()
    try:
        resp = requests.get(f"{url}/models", timeout=4)
    except Exception as e:
        status.latency_ms = int((time.time() - started) * 1000)
        if _is_connection_error(e):
            status.state = STATE_OFFLINE
            status.detail = f"Connected to {url} but /models did not answer."
            status.fix = "Confirm the Local Server URL in Settings -> Local AI."
        else:
            status.state = STATE_ERROR
            status.detail = f"Local server error: {str(e)[:200]}"
        return status

    status.latency_ms = int((time.time() - started) * 1000)

    if resp.status_code != 200:
        status.state = STATE_ERROR
        status.detail = f"Local server /models returned HTTP {resp.status_code}."
        status.fix = "Confirm the Local Server URL in Settings -> Local AI."
        return status

    try:
        ids = [m.get("id", "") for m in resp.json().get("data", [])]
        ids = [i for i in ids if i]
    except Exception:
        ids = []

    status.extra["available_models"] = ids

    if not ids:
        status.state = STATE_ERROR
        status.detail = "Local server is up but has no models loaded."
        status.fix = "Pull a model first, e.g. 'ollama pull llama3.2'."
        return status

    # Ollama reports "llama3.2:latest" for a model the user configures as
    # "llama3.2", so compare on the bare name as well.
    def _base(s: str) -> str:
        return s.split(":")[0].strip().lower()

    if model not in ids and _base(model) not in {_base(i) for i in ids}:
        status.state = STATE_ERROR
        status.detail = (
            f"Local server is up, but the configured model '{model}' is not "
            f"loaded. Available: {', '.join(ids[:5])}"
        )
        status.fix = "Set Local Model Name in Settings -> Local AI to one of the above."
        return status

    status.state = STATE_OK
    status.detail = f"Serving '{model}' on {url}."
    return status


# ── Cache + aggregate ────────────────────────────────────────────────────────

PROBES: dict[str, Callable[[bool], ProviderStatus]] = {
    "Gemini": probe_gemini,
    "OpenRouter": probe_openrouter,
    "Local": probe_local,
}

_cache: dict[str, tuple[float, ProviderStatus]] = {}
_cache_lock = threading.RLock()


def _ttl(deep: bool) -> float:
    return DEEP_TTL if deep else SHALLOW_TTL


def check(name: str, deep: bool = True, force: bool = False) -> ProviderStatus:
    """Check one named provider, honouring the cache unless `force`."""
    probe = PROBES.get(name)
    if probe is None:
        return ProviderStatus(name, STATE_ERROR, f"Unknown provider '{name}'.")

    now = time.time()
    with _cache_lock:
        hit = _cache.get(name)
        if hit and not force and (now - hit[0]) < _ttl(deep):
            return hit[1]

    try:
        status = probe(deep)
    except Exception as e:  # a probe must never take the app down
        logger.exception(f"[Health] probe for {name} crashed")
        status = ProviderStatus(name, STATE_ERROR, f"Health check failed: {str(e)[:200]}")

    status.checked_at = time.time()
    with _cache_lock:
        _cache[name] = (status.checked_at, status)
    return status


def check_all(deep: bool = True, force: bool = False) -> dict[str, ProviderStatus]:
    return {name: check(name, deep=deep, force=force) for name in PROBES}


def any_working(deep: bool = False, force: bool = False) -> bool:
    """True when at least one provider is expected to answer a real request."""
    return any(s.usable for s in check_all(deep=deep, force=force).values())


def working_providers(deep: bool = False, force: bool = False) -> list[str]:
    return [n for n, s in check_all(deep=deep, force=force).items() if s.usable]


def preferred_provider() -> Optional[str]:
    """The provider the user selected, if it is actually usable.

    Returns None when the configured default is down, which is the signal the
    caller needs in order to fail over instead of hammering a dead provider.
    """
    configured = (_load_settings().get("default_ai_provider") or "OpenRouter").strip()
    alias = {"Google Gemini": "Gemini", "gemini": "Gemini"}.get(configured, configured)
    status = check(alias)
    if status.usable:
        return alias
    return None


def unavailable_note(exc: Optional[BaseException] = None) -> str:
    """One actionable line naming the thing to go fix.

    Prefers live probe results (they cover every provider at once) and falls
    back to parsing the exception text when a probe has not run yet.
    """
    statuses = check_all(deep=False)

    blocking = [s for s in statuses.values() if s.state in
                (STATE_QUOTA, STATE_INVALID_KEY, STATE_MISSING)]
    if blocking:
        worst = blocking[0]
        fix = worst.fix or "Check Settings -> AI Providers."
        return f"{worst.name}: {worst.detail} {fix}".strip()

    offline = [s for s in statuses.values() if s.state == STATE_OFFLINE]
    if offline and len(offline) == len(statuses):
        return "No AI provider is reachable. Check your internet connection."

    if exc is not None:
        text = str(exc)
        if _is_daily_quota_text(text):
            return ("OpenRouter's free-model daily limit is used up. Add credits "
                    "at openrouter.ai/credits, or add a free Gemini AI Studio key "
                    "(aistudio.google.com/apikey).")
        if "API key is missing" in text or "Authentication failed" in text:
            return "The AI provider key is missing or invalid. Check Settings -> AI Providers."
        if "Connection" in text or "Timeout" in text:
            return "Could not reach the AI provider. Check your internet connection."

    return "The AI provider is unavailable right now. Please try again shortly."


def report(deep: bool = True) -> str:
    """Human-readable report, shared by the CLI and the UI panel."""
    lines: list[str] = []
    statuses = check_all(deep=deep, force=True)
    for name, s in statuses.items():
        lines.append(s.one_line())
        if s.fix:
            lines.append(f"    → {s.fix}")
        if s.extra.get("available_models"):
            lines.append(f"    → local models: {', '.join(s.extra['available_models'][:8])}")
    usable = [n for n, s in statuses.items() if s.usable]
    lines.append("")
    if usable:
        lines.append(f"Answerable via: {', '.join(usable)}")
    else:
        lines.append("NO PROVIDER CAN ANSWER RIGHT NOW.")
        lines.append("The assistant will queue requests and answer once one recovers.")
    return "\n".join(lines)


def invalidate() -> None:
    with _cache_lock:
        _cache.clear()


if __name__ == "__main__":
    print("=" * 62)
    print("  Brahma Evo — AI Provider Health Check")
    print("=" * 62)
    print()
    print(report(deep=True))