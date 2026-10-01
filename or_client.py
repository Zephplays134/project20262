from core.user_paths import get_user_data_dir
import json
import sys
import time
import base64
import logging
import threading
from pathlib import Path
from typing import Optional

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("openrouter_client")

def _get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent


BASE_DIR     = _get_base_dir()
API_KEY_PATH = get_user_data_dir() / "config" / "api_keys.json"

def _load_api_key() -> str:
    try:
        with open(API_KEY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        key = data.get("openrouter_api_key", "").strip()
        return key
    except FileNotFoundError:
        return ""
    except Exception as e:
        logger.warning(f"[OpenRouter] Failed to load API key: {e}")
        return ""

TEXT_MODELS: list[str] = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "minimax/minimax-m2.5:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "qwen/qwen3-coder:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-3-27b-it:free",
    "arcee-ai/trinity-large-preview:free",
    "z-ai/glm-4.5-air:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
    "google/gemma-3-12b-it:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "nvidia/nemotron-nano-9b-v2:free",
    "google/gemma-3-4b-it:free",
    "google/gemma-3n-e4b-it:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "google/gemma-3n-e2b-it:free",
    "liquid/lfm-2.5-1.2b-instruct:free",
    "liquid/lfm-2.5-1.2b-thinking:free",
]

VISION_MODELS: list[str] = [
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "nvidia/llama-nemotron-embed-vl-1b-v2:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-3n-e4b-it:free",
    "google/gemma-3n-e2b-it:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
]

API_URL               = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MAX_TOKENS    = 4096
DEFAULT_TEMPERATURE   = 0.7
REQUEST_TIMEOUT       = 60   # seconds per request
MAX_RETRIES_PER_MODEL = 2    # attempts before moving to next model
RETRY_DELAY           = 2    # seconds between retries
RATE_LIMIT_COOLDOWN   = 60   # seconds before retrying a rate-limited model

_rate_limited: dict[str, float] = {}

# ── Dead-model quarantine ────────────────────────────────────────────────────
#
# OpenRouter retires free models constantly. A retired slug still answers the
# request — with HTTP 404 — so an unfiltered pool costs 2 attempts x N dead
# slugs before reaching a live one. Measured on this machine: 17.6 s on the
# first call and 54.3 s on the next, for a one-line answer.
#
# A 404 means the slug is gone, not that the request was unlucky, so retrying it
# is pure waste. Retired slugs are remembered for DEAD_MODEL_TTL and skipped
# without a request until that window passes. Only 404/403 quarantine; 429 is
# per-account load and handled by the existing cooldown instead.
_dead_models: dict[str, float] = {}
DEAD_MODEL_TTL = 6 * 60 * 60   # re-check a retired slug after 6 hours

_live_models_cache: list[str] = []
_live_models_cache_ts: float = 0.0
_live_models_lock = threading.Lock()
_LIVE_MODELS_TTL = 30 * 60


def _is_dead(model: str) -> bool:
    ts = _dead_models.get(model)
    if ts is None:
        return False
    if time.time() - ts > DEAD_MODEL_TTL:
        del _dead_models[model]
        return False
    return True


def _mark_dead(model: str) -> None:
    _dead_models[model] = time.time()
    logger.info(f"[OpenRouter] Quarantined retired model: {model}")


class DailyQuotaExhausted(RuntimeError):
    """The account's free-model daily allowance is used up.

    This is account-wide, not model-specific: every ":free" slug returns the same
    429. Walking the rest of the pool cannot help, so the walk stops here and the
    user is told what actually happened — "all models failed" hides this entirely.
    """


def _is_daily_quota_error(text: str) -> bool:
    low = (text or "").lower()
    return "free-models-per-day" in low or "free models per day" in low \
        or "daily limit" in low and "free" in low

class OpenRouterClient:

    def __init__(self) -> None:
        self.api_key  = _load_api_key()
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "https://github.com/brahma-ai",
            "X-Title":       "Brahma Evo",
        }

    def _is_rate_limited(self, model: str) -> bool:
        ts = _rate_limited.get(model)
        if ts is None:
            return False
        if time.time() - ts > RATE_LIMIT_COOLDOWN:
            del _rate_limited[model]
            return False
        return True

    def _mark_rate_limited(self, model: str) -> None:
        _rate_limited[model] = time.time()
        logger.warning(
            f"[OpenRouter] Rate limited: {model} — "
            f"cooling down for {RATE_LIMIT_COOLDOWN}s"
        )

    def _call(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        response_format: Optional[dict] = None,
    ) -> Optional[str]:
        payload: dict = {
            "model":       model,
            "messages":    messages,
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }
        if response_format:
            payload["response_format"] = response_format

        if not self.api_key:
            raise PermissionError(
                "[OpenRouter] API key is missing. Add a valid sk-or- key in config/api_keys.json."
            )

        for attempt in range(1, MAX_RETRIES_PER_MODEL + 1):
            try:
                resp = requests.post(
                    API_URL,
                    headers=self._headers,
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                )

                if resp.status_code == 401:
                    raise PermissionError(
                        f"[OpenRouter] Authentication failed for model {model}. "
                        "Check your API key in config/api_keys.json."
                    )

                if resp.status_code == 403:
                    # "only available on agentic ..." is a property of the slug,
                    # not the account, so quarantine it instead of retrying.
                    if "only available on" in (resp.text or "").lower():
                        _mark_dead(model)
                        return None
                    raise PermissionError(
                        f"[OpenRouter] Access denied for model {model} (HTTP 403). "
                        "Check your account permissions and model access."
                    )

                if resp.status_code == 404:
                    # Retired or unavailable slug. Retrying cannot help, so
                    # quarantine it and move on immediately.
                    _mark_dead(model)
                    return None

                if resp.status_code == 429:
                    # A daily free-tier allowance is an account-level wall;
                    # retrying other free slugs just repeats it.
                    if _is_daily_quota_error(resp.text):
                        raise DailyQuotaExhausted(
                            "OpenRouter's free-model daily limit for this account is "
                            "used up (free-models-per-day). Add credits at "
                            "https://openrouter.ai/credits, or wait for the daily reset. "
                            "A Google AI Studio key also works and is free."
                        )
                    self._mark_rate_limited(model)
                    return None

                if resp.status_code == 200:
                    data    = resp.json()
                    content = (
                        data.get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "")
                    )
                    return content.strip() if content else None

                logger.warning(
                    f"[OpenRouter] {model} → HTTP {resp.status_code} "
                    f"(attempt {attempt}/{MAX_RETRIES_PER_MODEL})"
                )

            except requests.exceptions.Timeout:
                logger.warning(
                    f"[OpenRouter] {model} → Timeout "
                    f"(attempt {attempt}/{MAX_RETRIES_PER_MODEL})"
                )
            except PermissionError:
                raise
            except DailyQuotaExhausted:
                raise
            except Exception as e:
                logger.error(f"[OpenRouter] {model} → Unexpected error: {e}")

            if attempt < MAX_RETRIES_PER_MODEL:
                time.sleep(RETRY_DELAY)

        return None

    def _live_free_models(self, limit: int = 8) -> list[str]:
        """Fetches currently-available free models from the OpenRouter API.

        The hardcoded TEXT_MODELS list goes stale as providers deprecate slugs;
        this keeps the fallback pool alive. Cached for 30 minutes because a
        failed lookup here means walking the (slow) hardcoded pool instead.
        """
        global _live_models_cache
        with _live_models_lock:
            if _live_models_cache and (time.time() - _live_models_cache_ts) < _LIVE_MODELS_TTL:
                return _live_models_cache[:limit]
            try:
                resp = requests.get(
                    "https://openrouter.ai/api/v1/models",
                    headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
                    timeout=15,
                )
                if resp.status_code != 200:
                    return _live_models_cache[:limit]
                data = resp.json().get("data", [])
                free_ids = [
                    m["id"] for m in data
                    if ":free" in m.get("id", "")
                    and m.get("context_length", 0) >= 32000
                ]
                _live_models_cache = free_ids
                globals()["_live_models_cache_ts"] = time.time()
                return free_ids[:limit]
            except Exception as e:
                logger.warning(f"[OpenRouter] Could not fetch live model list: {e}")
                return _live_models_cache[:limit]

    def _call_with_fallback(
        self,
        pool: list[str],
        messages: list[dict],
        model: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        response_format: Optional[dict] = None,
    ) -> str:
        if not self.api_key:
            # Fail before walking the pool: without a key every model returns
            # the same 401, so the walk is pure latency on the way to an error.
            raise PermissionError(
                "[OpenRouter] API key is missing. Add a valid sk-or- key in config/api_keys.json."
            )

        if model and not self._is_rate_limited(model):
            try:
                result = self._call(model, messages, max_tokens, temperature, response_format)
                if result:
                    return result
                logger.info(
                    f"[OpenRouter] Requested model failed, "
                    f"falling back to pool: {model}"
                )
            except PermissionError:
                raise

        # Live models first: the hardcoded pool is mostly retired slugs, and each
        # one costs a round trip (2 attempts) before the next is tried. A model
        # the API just listed is far more likely to answer than one on this list.
        ordered = [m for m in self._live_free_models() if not _is_dead(m)]
        ordered += [m for m in pool if m not in ordered and not _is_dead(m)]

        for m in ordered:
            if self._is_rate_limited(m):
                continue
            logger.info(f"[OpenRouter] Trying: {m}")
            result = self._call(m, messages, max_tokens, temperature, response_format)
            if result:
                logger.info(f"[OpenRouter] ✓ Success: {m}")
                return result

        raise RuntimeError(
            "[OpenRouter] All models failed or are rate-limited. "
            "Check your API key and network connection."
        )

    def chat(
        self,
        prompt: str,
        system: str = (
            "You are a component of Brahma Evo, an autonomous self-evolving personal assistant. "
            "Be concise, helpful, and precise."
        ),
        history: Optional[list[dict]] = None,
        model: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        messages = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        return self._call_with_fallback(
            TEXT_MODELS, messages, model, max_tokens, temperature
        )

    def chat_json(
        self,
        prompt: str,
        system: str = (
            "Return ONLY valid JSON. "
            "No markdown fences, no extra text, no explanation."
        ),
        model: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> dict:
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ]
        raw = self._call_with_fallback(
            TEXT_MODELS, messages, model, max_tokens, temperature=0.2
        )

        clean = raw.strip()
        if clean.startswith("```"):
            parts = clean.split("```")
            clean = parts[1] if len(parts) > 1 else clean
            if clean.startswith("json"):
                clean = clean[4:]
        clean = clean.strip().rstrip("`").strip()

        try:
            return json.loads(clean)
        except json.JSONDecodeError as e:
            logger.error(
                f"[OpenRouter] JSON parse failed: {e}\n"
                f"Raw response (first 300 chars): {raw[:300]}"
            )
            raise ValueError(
                f"Model returned unparseable JSON: {e}\n"
                f"Raw output: {raw[:200]}"
            )

    def vision(
        self,
        prompt: str,
        image_b64: str,
        mime: str = "image/png",
        system: str = "Analyze the image and describe what you see clearly and concisely.",
        model: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> str:
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{image_b64}"
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        return self._call_with_fallback(
            VISION_MODELS, messages, model, max_tokens, temperature=0.2
        )

    def vision_from_file(
        self,
        prompt: str,
        image_path: str,
        system: str = "Analyze the image and describe what you see clearly and concisely.",
        model: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> str:
        path = Path(image_path)
        mime_map = {
            ".png":  "image/png",
            ".jpg":  "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif":  "image/gif",
        }
        mime = mime_map.get(path.suffix.lower(), "image/png")

        with open(path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        return self.vision(prompt, image_b64, mime, system, model, max_tokens)

    def multi_turn(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
    
        return self._call_with_fallback(
            TEXT_MODELS, messages, model, max_tokens, temperature
        )

    def available_models(self) -> dict:
        return {
            "text_models":   TEXT_MODELS,
            "vision_models": VISION_MODELS,
            "rate_limited":  list(_rate_limited.keys()),
            "total_text":    len(TEXT_MODELS),
            "total_vision":  len(VISION_MODELS),
        }

client = OpenRouterClient()

if __name__ == "__main__":
    print("=" * 55)
    print("  Brahma Evo — OpenRouter Client Self-Test")
    print("=" * 55)

    print("\n[TEST 1] Basic chat...")
    try:
        reply = client.chat("Introduce yourself in one sentence.")
        print(f"  Response : {reply}")
        print(f"  Status   : PASS ✓")
    except Exception as e:
        print(f"  Status   : FAIL ✗ — {e}")

    print("\n[TEST 2] JSON mode...")
    try:
        data = client.chat_json(
            'List 3 programming languages. Format: {"languages": ["a", "b", "c"]}',
            system="Return only valid JSON. No extra text."
        )
        print(f"  Response : {data}")
        print(f"  Status   : PASS ✓")
    except Exception as e:
        print(f"  Status   : FAIL ✗ — {e}")

    print("\n[TEST 3] Multi-turn conversation...")
    try:
        history = [
            {"role": "system",    "content": "You are a helpful assistant. Be brief."},
            {"role": "user",      "content": "My name is User."},
            {"role": "assistant", "content": "Hello User, how can I help you?"},
            {"role": "user",      "content": "What is my name?"},
        ]
        reply = client.multi_turn(history)
        print(f"  Response : {reply}")
        print(f"  Status   : PASS ✓")
    except Exception as e:
        print(f"  Status   : FAIL ✗ — {e}")

    print("\n[TEST 4] Model pool info...")
    info = client.available_models()
    print(f"  Text models   : {info['total_text']}")
    print(f"  Vision models : {info['total_vision']}")
    print(f"  Rate limited  : {info['rate_limited'] or 'none'}")
    print(f"  Status        : PASS ✓")

    print("\n" + "=" * 55)
    print("  All tests complete.")
    print("=" * 55)
