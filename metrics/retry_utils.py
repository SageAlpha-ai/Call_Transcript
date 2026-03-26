from __future__ import annotations

import logging
import random
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

MAX_RETRIES = 3
BASE_DELAY_S = 0.4


def retry_sync(
    fn: Callable[[], T],
    *,
    max_retries: int = MAX_RETRIES,
    operation: str = "operation",
) -> T:
    last_exc: BaseException | None = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries - 1:
                logger.exception(
                    "%s failed after %d attempts",
                    operation,
                    max_retries,
                )
                raise
            delay = BASE_DELAY_S * (2**attempt) + random.uniform(0, 0.15)
            logger.warning(
                "%s attempt %d/%d failed: %s; retrying in %.2fs",
                operation,
                attempt + 1,
                max_retries,
                exc,
                delay,
            )
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc
