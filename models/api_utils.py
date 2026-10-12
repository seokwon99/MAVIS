"""Retry helper shared by the API backends (OpenAI, Anthropic)."""
import logging
import time

logger = logging.getLogger(__name__)


def call_with_retry(fn, max_tries=5, base_delay=1.0, **kwargs):
    """Call `fn(**kwargs)` with exponential backoff; return "Error" (and log it) once all tries fail."""
    for attempt in range(1, max_tries + 1):
        try:
            return fn(**kwargs)
        except Exception as e:  # noqa: BLE001
            if attempt == max_tries:
                logger.warning("API call failed after %d tries: %r", max_tries, e)
                return "Error"
            delay = base_delay * (2 ** (attempt - 1))
            logger.info("API call failed (try %d/%d): %r; retrying in %.1fs", attempt, max_tries, e, delay)
            time.sleep(delay)
