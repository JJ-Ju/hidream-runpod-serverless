import handler


class FakeService:
    def handle_job(self, job, progress_callback=None):
        assert progress_callback is not None
        return {"image_url": "https://cdn.example.com/out.png", "job_id": job["id"]}


def test_handler_delegates_to_singleton_service(monkeypatch):
    monkeypatch.setattr(handler, "get_service", lambda: FakeService())

    result = handler.handler({"id": "job-1", "input": {"prompt": "A red biplane"}})

    assert result == {"image_url": "https://cdn.example.com/out.png", "job_id": "job-1"}
