from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable


ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass
class ProgressReporter:
    callback: ProgressCallback | None = None
    started_at: float = field(default_factory=time.monotonic)
    generation_started_at: float | None = None

    def emit(
        self,
        stage: str,
        percent: float,
        message: str,
        **extra: Any,
    ) -> None:
        if self.callback is None:
            return
        payload = {
            "stage": stage,
            "percent": max(0.0, min(100.0, round(percent, 2))),
            "message": message,
            "elapsed_seconds": round(time.monotonic() - self.started_at, 2),
        }
        payload.update({key: value for key, value in extra.items() if value is not None})
        try:
            self.callback(payload)
        except Exception:
            pass

    def generation_step(self, step_index: int, total_steps: int, preview_fn=None) -> None:
        if total_steps <= 0:
            return
        if self.generation_started_at is None:
            self.generation_started_at = time.monotonic()

        completed_steps = min(step_index + 1, total_steps)
        generation_fraction = completed_steps / total_steps
        elapsed = max(time.monotonic() - self.generation_started_at, 0.001)
        seconds_per_step = elapsed / completed_steps
        remaining_steps = max(total_steps - completed_steps, 0)
        eta_seconds = round(seconds_per_step * remaining_steps, 2)
        percent = 20.0 + generation_fraction * 70.0

        self.emit(
            "generating",
            percent,
            f"Generating step {completed_steps}/{total_steps}",
            step=completed_steps,
            total_steps=total_steps,
            eta_seconds=eta_seconds,
        )


def runpod_progress_callback(job: dict[str, Any]) -> ProgressCallback:
    def callback(payload: dict[str, Any]) -> None:
        try:
            import runpod

            runpod.serverless.progress_update(job, json.dumps(payload, separators=(",", ":")))
        except Exception:
            pass

    return callback

