"""Hold requests while every AI provider is down, answer them when one recovers.

The user asked for the third degradation mode: when no provider can answer, do
not fail the interaction and do not fake a reply — park the request and deliver
the real answer as soon as something works. Quotas reset daily, keys get fixed,
Ollama gets started; none of those are knowable in advance, so the only honest
behaviour is to keep the request and retry on a timer.

Design notes
------------
* One daemon worker, started lazily, polls `provider_health` on an interval.
* Bounded: at most `max_items` requests are held, each for at most `item_ttl`.
  Without this a long outage turns the queue into an unbounded memory leak that
  then fires a burst of stale replies at the user when service returns.
* `resolver` and `deliver` are injected, so this module has no opinion about
  which provider to use or how the UI is driven. `deliver` is called on the
  worker thread.
* Backoff: while nothing works the poll interval grows to `max_poll_interval`,
  so a long outage costs almost nothing and recovery is still noticed promptly.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import provider_health

logger = logging.getLogger("pending_queue")


@dataclass
class PendingRequest:
    prompt: str
    system: str = ""
    label: str = "request"
    created_at: float = field(default_factory=time.time)
    attempts: int = 0


class PendingQueue:
    """FIFO queue of unanswered requests, retried until a provider recovers."""

    def __init__(
        self,
        resolver: Callable[[str, str], str],
        deliver: Callable[[str, PendingRequest], None],
        on_recovered: Optional[Callable[[list], None]] = None,
        poll_interval: float = 20.0,
        max_poll_interval: float = 300.0,
        max_items: int = 25,
        item_ttl: float = 6 * 60 * 60,
    ) -> None:
        self._resolve = resolver
        self._deliver = deliver
        self._on_recovered = on_recovered
        self._poll = poll_interval
        self._max_poll = max_poll_interval
        self._max_items = max_items
        self._item_ttl = item_ttl

        self._items: list[PendingRequest] = []
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._last_error: str = ""

    # ── Public API ───────────────────────────────────────────────────────────

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._items)

    @property
    def last_error(self) -> str:
        return self._last_error

    def is_queueing(self) -> bool:
        """True when the queue is active — used to explain the wait to the user."""
        return self._thread is not None and self._thread.is_alive()

    def enqueue(self, prompt: str, system: str = "", label: str = "request") -> bool:
        """Park a request. Returns False if it was dropped (queue full).

        The queue starts by itself on first use, so callers do not have to
        sequence a start() during boot.
        """
        self._ensure_worker()
        with self._lock:
            if len(self._items) >= self._max_items:
                # Drop the oldest rather than the newest: during a long outage
                # the newest request is the one the user still cares about.
                dropped = self._items.pop(0)
                logger.warning(
                    f"[PendingQueue] Queue full, dropped oldest: {dropped.label}"
                )
            self._items.append(PendingRequest(prompt=prompt, system=system, label=label))
            logger.info(
                f"[PendingQueue] Queued '{label}' "
                f"({len(self._items)}/{self._max_items} held). "
                f"Last error: {self._last_error[:120]}"
            )
            return True

    def flush(self) -> list[str]:
        """Try everything right now. Returns the labels that got answered."""
        answered = self._drain()
        if answered and self._on_recovered:
            try:
                self._on_recovered(answered)
            except Exception:
                logger.exception("[PendingQueue] on_recovered callback failed")
        return answered

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)
        self._thread = None

    # ── Internals ────────────────────────────────────────────────────────────

    def _ensure_worker(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._worker, name="brahma-pending-queue", daemon=True
        )
        self._thread.start()

    def _take_expired(self) -> None:
        now = time.time()
        with self._lock:
            self._items = [i for i in self._items if now - i.created_at < self._item_ttl]

    def _drain(self) -> list[str]:
        """Attempt every held request once. Returns labels that succeeded."""
        self._take_expired()
        with self._lock:
            batch = list(self._items)

        answered: list[str] = []
        for item in batch:
            item.attempts += 1
            try:
                answer = self._resolve(item.prompt, item.system)
            except Exception as e:
                self._last_error = str(e)
                continue
            if not answer or not str(answer).strip():
                continue

            with self._lock:
                if item in self._items:
                    self._items.remove(item)
            answered.append(item.label)
            logger.info(f"[PendingQueue] Delivered queued '{item.label}'")
            try:
                self._deliver(str(answer).strip(), item)
            except Exception:
                logger.exception(
                    f"[PendingQueue] Delivery of '{item.label}' failed; answer dropped"
                )
        return answered

    def _worker(self) -> None:
        delay = self._poll
        while not self._stop.is_set():
            if self._stop.wait(delay):
                break

            with self._lock:
                if not self._items:
                    delay = self._poll
                    continue

            # Shallow probe: key shape and reachability, no generation cost.
            # A provider that reports itself usable is then actually tried, and
            # a genuine failure just means we fall back to waiting.
            try:
                ready = provider_health.any_working(deep=False)
            except Exception:
                logger.exception("[PendingQueue] Health check failed")
                ready = False

            if not ready:
                delay = min(delay * 1.5, self._max_poll)
                continue

            answered = self._drain()
            if answered:
                provider_health.invalidate()
                delay = self._poll
                if self._on_recovered:
                    try:
                        self._on_recovered(answered)
                    except Exception:
                        logger.exception("[PendingQueue] on_recovered callback failed")
            else:
                # Health looked fine but the real call still failed — back off
                # rather than spinning on a provider that keeps rejecting us.
                delay = min(delay * 1.5, self._max_poll)


# ── Process-wide queue ───────────────────────────────────────────────────────
#
# The app talks to one queue from several places (the chat handler, the UI
# pending counter, the dev agent), so it is configured once at startup and
# reached through `get_queue()`. Callers must tolerate None: the queue is a
# safety net, and nothing should crash when it has not been wired up.

_default_queue: Optional[PendingQueue] = None


def configure(
    resolver: Callable[[str, str], str],
    deliver: Callable[[str, PendingRequest], None],
    on_recovered: Optional[Callable[[list], None]] = None,
    **kwargs,
) -> PendingQueue:
    """Create and install the process-wide queue, replacing any previous one."""
    global _default_queue
    if _default_queue is not None:
        try:
            _default_queue.stop()
        except Exception:
            pass
    _default_queue = PendingQueue(
        resolver=resolver,
        deliver=deliver,
        on_recovered=on_recovered,
        **kwargs,
    )
    return _default_queue


def get_queue() -> Optional[PendingQueue]:
    return _default_queue


def enqueue(prompt: str, system: str = "", label: str = "request") -> bool:
    """Park a request on the process-wide queue.

    Returns False when no queue is configured, so the caller can fall back to
    telling the user directly.
    """
    q = _default_queue
    if q is None:
        logger.info(f"[PendingQueue] Dropped '{label}': queue not configured.")
        return False
    return q.enqueue(prompt, system=system, label=label)