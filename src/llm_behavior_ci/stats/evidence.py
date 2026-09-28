from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol


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


class Detector(Protocol):
    def update(self, observation: float) -> Evidence: ...

    def reset(self) -> None: ...

    def snapshot(self) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class PairedSuccess:
    candidate: float
    reference: float
