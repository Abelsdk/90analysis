"""Download a URL once and keep the response on disk."""

import time
from pathlib import Path

import requests

USER_AGENT = "90analysis (+https://github.com/Abelsdk/90analysis)"


class SourceUnavailable(RuntimeError):
    """The source kept failing after all retries."""


def fetch_text(
    url: str,
    cache_path: Path,
    *,
    max_age_hours: float | None = None,
    retries: int = 3,
    backoff_seconds: float = 2.0,
) -> str:
    """Return the body of `url`, reading `cache_path` instead when it is fresh enough.

    `max_age_hours=None` means the cached copy never expires (use it for past data).
    """
    if cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if max_age_hours is None or age_hours < max_age_hours:
            return cache_path.read_text(encoding="utf-8")

    last_error = ""
    for attempt in range(retries):
        try:
            response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        except requests.RequestException as error:
            last_error = str(error)
        else:
            if response.ok and response.content:
                text = response.content.decode("utf-8-sig", errors="replace")
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(text, encoding="utf-8")
                return text
            last_error = f"HTTP {response.status_code}"
            if 400 <= response.status_code < 500:
                break  # a client error won't fix itself on retry
        time.sleep(backoff_seconds * (attempt + 1))

    raise SourceUnavailable(f"{url} failed after {retries} attempts: {last_error}")
