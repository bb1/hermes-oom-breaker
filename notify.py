"""Debug notices: opt-in one-line Telegram posts for operator visibility.

Implements the h3 delivery pattern: direct Telegram Bot API via stdlib
urllib, with the profile's ``.env`` supplying ``TELEGRAM_BOT_TOKEN`` /
``TELEGRAM_HOME_CHANNEL`` — no gateway, no hermes runtime, so it works
while the LLM brain is down. Hermes-free at import: the hermes home
resolver imports hermes lazily inside its body. The module never raises
— every failure path is a loud warning and a dropped post — and the
synchronous half of ``post()`` is a lock plus a daemon-thread spawn
only: no I/O on the caller's thread.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.request
from typing import Callable, Dict, Optional, Tuple

from .config import Config

logger = logging.getLogger("oom_breaker")

_BOT_API = "https://api.telegram.org"


def _default_home_fn() -> Optional[str]:
    """The profile's hermes home (lazy hermes import so this module's
    import stays hermes-free); any failure — import or call — yields
    None, so the post is dropped with a warning."""
    try:
        from hermes_constants import get_hermes_home
        return get_hermes_home()
    except Exception:
        return None


def _load_env(home: str) -> Dict[str, str]:
    """Parse ``home/.env`` (``KEY=VALUE`` lines; blanks and ``#`` comments
    skipped; one pair of matching outer quotes stripped). A missing or
    unreadable file yields {}; never raises."""
    env: Dict[str, str] = {}
    path = os.path.join(home, ".env")
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                    value = value[1:-1]
                if key:
                    env[key] = value
    except Exception:
        pass
    return env


def _send(token: str, chat: str, thread: Optional[str], text: str) -> bool:
    """POST one message to the Telegram Bot API (20 s timeout) and return
    the response's ``ok`` field. ``message_thread_id`` is attached only
    when a thread is given and all-digits. Any exception — payload build,
    network, HTTP, bad JSON — yields False. Never raises."""
    try:
        payload: Dict[str, object] = {"chat_id": chat, "text": text}
        if thread is not None and thread.isdigit():
            payload["message_thread_id"] = int(thread)
        req = urllib.request.Request(
            f"{_BOT_API}/bot{token}/sendMessage",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST")
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read().decode("utf-8", "replace"))
        return bool(body.get("ok", False))
    except Exception:
        return False


def _spawn(fn: Callable[[], None]) -> None:
    """Run ``fn`` in a daemon thread (the caller never blocks)."""
    threading.Thread(target=fn, daemon=True).start()


def _parse_channel(channel: str) -> Optional[Tuple[str, Optional[str]]]:
    """Parse ``telegram:<chat_id>[:<thread_id>]`` -> (chat_id,
    thread_id|None); a wrong prefix or a missing chat yields None."""
    parts = channel.split(":")
    if len(parts) < 2 or parts[0] != "telegram" or not parts[1]:
        return None
    return parts[1], (parts[2] if len(parts) > 2 and parts[2] else None)


class Notifier:
    """Opt-in debug notices: one rate-limited Telegram post per event type.

    The notifier stays dumb — it sends the fully composed ``text`` as-is —
    and never raises: a failed post is a loud warning. ``post()`` itself
    only rate-limits under a lock (one bucket per event type) and spawns
    the real work (home/env/target resolution, the Bot API call) onto a
    daemon thread; a total no-op while ``cfg.debug`` is off.
    """

    def __init__(self, cfg: Config, *,
                 now: Callable[[], float] = time.time,
                 home_fn: Optional[Callable[[], Optional[str]]] = None,
                 env_loader: Optional[Callable[[str], Dict[str, str]]] = None,
                 sender: Optional[Callable[[str, str, Optional[str], str], bool]] = None,
                 spawner: Optional[Callable[[Callable[[], None]], None]] = None) -> None:
        self._cfg = cfg
        self._now = now
        self._home_fn = home_fn if home_fn is not None else _default_home_fn
        self._env_loader = env_loader if env_loader is not None else _load_env
        self._sender = sender if sender is not None else _send
        self._spawner = spawner if spawner is not None else _spawn
        self._last: Dict[str, float] = {}
        self._lock = threading.Lock()

    def post(self, event: str, text: str, *, home: Optional[str] = None) -> None:
        """Rate-limit and spawn one debug post for ``event`` (a rate bucket
        per event type). Never raises, never blocks; a total no-op while
        ``cfg.debug`` is off."""
        if not self._cfg.debug:
            return
        try:
            with self._lock:
                t = self._now()
                last = self._last.get(event)
                # No prior stamp is never rate-limited (a first post goes out
                # even at clock t=0; the .get(...) default 0.0 would drop it).
                if last is not None and t - last < self._cfg.debug_min_interval:
                    return
                self._last[event] = t      # stamp at acceptance
            self._spawner(lambda: self._post(event, text, home))
        except Exception:
            # The injected seams (clock/spawner) are the only uncontained
            # path; a failure there drops the post loudly instead of
            # escaping into the caller's hook.
            logger.warning("oom-breaker debug: post for event %r failed",
                           event, exc_info=True)

    def _post(self, event: str, text: str, home: Optional[str]) -> None:
        """The spawned work: resolve home/env/target and send. Every
        failure path is a loud warning (and a dropped post); never raises
        (the token is never logged)."""
        try:
            home = home or self._home_fn()
            if not home:
                logger.warning(
                    "oom-breaker debug: no hermes home for event %r; post "
                    "dropped", event)
                return
            env = self._env_loader(home)
            if self._cfg.debug_channel:
                target = _parse_channel(self._cfg.debug_channel)
                if target is None:
                    logger.warning(
                        "oom-breaker debug: invalid debug_channel %r; post "
                        "for event %r dropped", self._cfg.debug_channel, event)
                    return
                chat, thread = target
            else:
                chat = env.get("TELEGRAM_HOME_CHANNEL", "")
                thread = None
            token = env.get("TELEGRAM_BOT_TOKEN", "")
            if not token or not chat:
                logger.warning(
                    "oom-breaker debug: no bot token or home channel in "
                    "%s/.env; post for event %r dropped", home, event)
                return
            try:
                ok = self._sender(token, chat, thread, text)
            except Exception as exc:
                # A stdlib exception's text can carry the request URL (which
                # holds the bot token): only dump the traceback when the
                # exception's own message is proven token-free.
                safe = token not in str(exc)
                logger.warning(
                    "oom-breaker debug: sender failed for event %r: %s",
                    event, exc if safe else f"{type(exc).__name__} (redacted)",
                    exc_info=safe)
                return
            if ok:
                logger.info("oom-breaker debug: posted event %r", event)
            else:
                logger.warning(
                    "oom-breaker debug: post for event %r not delivered",
                    event)
        except Exception:
            logger.warning(
                "oom-breaker debug: post for event %r failed", event,
                exc_info=True)
