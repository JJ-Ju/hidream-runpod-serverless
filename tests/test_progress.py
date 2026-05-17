import json

from hidream_o1.progress import ProgressReporter, runpod_progress_callback


def test_progress_reporter_emits_generation_eta(monkeypatch):
    events = []
    ticks = iter([100.0, 102.0, 102.0])
    monkeypatch.setattr("hidream_o1.progress.time.monotonic", lambda: next(ticks))
    reporter = ProgressReporter(events.append, started_at=100.0)

    reporter.generation_step(0, 4)

    assert events == [
        {
            "stage": "generating",
            "percent": 37.5,
            "message": "Generating step 1/4",
            "elapsed_seconds": 2.0,
            "step": 1,
            "total_steps": 4,
            "eta_seconds": 6.0,
        }
    ]


def test_runpod_progress_callback_sends_json(monkeypatch):
    calls = []

    class FakeServerless:
        @staticmethod
        def progress_update(job, payload):
            calls.append((job, payload))

    class FakeRunPod:
        serverless = FakeServerless()

    monkeypatch.setitem(__import__("sys").modules, "runpod", FakeRunPod)
    job = {"id": "job-1"}

    runpod_progress_callback(job)({"stage": "validating", "percent": 2})

    assert calls[0][0] == job
    assert json.loads(calls[0][1]) == {"stage": "validating", "percent": 2}
