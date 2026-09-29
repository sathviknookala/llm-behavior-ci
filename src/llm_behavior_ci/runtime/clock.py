from __future__ import annotations

import time
from datetime import datetime, timezone


def wall_now() -> datetime:
    """UTC now that stays on the real clock while AppWorld freezes time.

    Opening a world starts freezegun, which replaces ``datetime.now``.
    Episode timestamps have to stay outside that freeze.
    """

    try:
        from freezegun.api import real_datetime
    except ImportError:
        return datetime.now(timezone.utc)
    return real_datetime.now(timezone.utc)


def monotonic() -> float:
    """``perf_counter`` that freezegun does not pin to the task datetime."""

    try:
        from freezegun.api import real_perf_counter
    except ImportError:
        return time.perf_counter()
    return real_perf_counter()
