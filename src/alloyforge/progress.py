from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable


@dataclass(slots=True)
class ProgressEvent:
    stage: str
    fraction: float
    message: str

    def as_dict(self) -> dict:
        return asdict(self)


ProgressCallback = Callable[[ProgressEvent], None]


class CancellationToken:
    def __init__(self, event=None):
        self._event = event

    @property
    def cancelled(self) -> bool:
        return bool(self._event is not None and self._event.is_set())

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise BuildCancelled("Model construction was cancelled")


class BuildCancelled(RuntimeError):
    pass
