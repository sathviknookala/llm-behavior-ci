from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

EVIDENCE_DIRECTIONS = frozenset(
    {"harmful", "beneficial", "insufficient", "undirected"}
)
"""The four readings a detector's evidence can carry.

``harmful`` and ``beneficial`` are only produced by a detector that tests a
paired candidate against a reference in a stated metric orientation.
``insufficient`` is that same detector's non-alarm state: it looked and
found neither. ``undirected`` is the default for a detector that does not
carry a harm/benefit distinction at all (a single-stream drift monitor,
for instance), so it is never read as harm evidence by a caller that
gates on direction.
"""


class StatisticsError(ValueError):
    pass


@dataclass(frozen=True)
class Evidence:
    method: str
    estimate: float
    sample_size: int
    alarm: bool
    boundary: float | None
    p_value: float | None
    details: tuple[tuple[str, float], ...]
    direction: str = "undirected"


class Detector(Protocol):
    def update(self, observation: float) -> Evidence: ...

    def reset(self) -> None: ...

    def snapshot(self) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class PairedSuccess:
    candidate: float
    reference: float
