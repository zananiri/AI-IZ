"""Progress lines for the corpus scripts (fetchers, downloads, vectorize.py).

One plain line every few seconds and one at the end -- no carriage returns, so it reads the same
in a terminal, in the run's .log file and in a Kaggle notebook's output:

    [12:04:31] laws: reading dump: 212.4/684.0 MB (31.1%) · 18.2 MB/s · ETA 26s · pages 51,204 · records 1,337
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime


def _duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


def _default_log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


class Progress:
    def __init__(self, label: str, total: float | None = None, unit: str = "", *,
                 log: Callable[[str], None] | None = None, every: float = 5.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.label, self.total, self.unit = label, total or None, unit
        self.count = 0
        self.extra: dict = {}
        self._log = log or _default_log
        self._every, self._clock = every, clock
        self._started = self._last = clock()

    def _amount(self, value: float) -> str:
        return f"{value / 1e6:,.1f}" if self.unit == "bytes" else f"{value:,.0f}"

    def line(self) -> str:
        elapsed = max(self._clock() - self._started, 1e-9)
        rate = self.count / elapsed
        unit = "MB" if self.unit == "bytes" else self.unit
        amount = self._amount(self.count) + (f"/{self._amount(self.total)}" if self.total else "")
        parts = [f"{self.label}: {amount} {unit}".rstrip()]
        if self.total:
            parts[0] += f" ({100 * self.count / self.total:.1f}%)"
        speed = f"{rate / 1e6:,.1f}" if self.unit == "bytes" else f"{rate:,.1f}"
        parts.append(f"{speed} {unit}/s" if unit else f"{speed}/s")
        if self.total and rate > 0 and self.count < self.total:
            parts.append(f"ETA {_duration((self.total - self.count) / rate)}")
        parts.append(f"elapsed {_duration(elapsed)}")
        parts += [f"{key} {value:,}" if isinstance(value, int) else f"{key} {value}" for key, value in self.extra.items()]
        return " · ".join(parts)

    def update(self, n: float = 1, **extra) -> None:
        self.count += n
        self.extra.update(extra)
        now = self._clock()
        if now - self._last >= self._every:
            self._last = now
            self._log(self.line())

    def set(self, value: float, **extra) -> None:
        self.update(value - self.count, **extra)

    def done(self, **extra) -> None:
        self.extra.update(extra)
        self._log(self.line() + " · done")
