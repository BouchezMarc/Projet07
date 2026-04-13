import importlib
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class ValidationError(Exception):
    pass


def _prepare_fake_rag_module() -> None:
    # Avoid external API/vector dependencies during API endpoint tests.
    fake_rag = types.ModuleType("app.rag")

    def fake_answer_question(question: str):
        return f"fake-answer:{question}", "fake-context"

    fake_rag.answer_question = fake_answer_question
    sys.modules["app.rag"] = fake_rag


def _load_api_module():
    # Use deterministic credentials for protected endpoint tests.
    os.environ.setdefault("REBUILD_USER", "admin")
    os.environ.setdefault("REBUILD_PASSWORD", "password")

    _prepare_fake_rag_module()
    return importlib.import_module("app.api")


def _test_health(client: TestClient) -> None:
    response = client.get("/")
    if response.status_code != 200:
        raise ValidationError(f"GET / failed: {response.status_code}")
    body = response.json()
    if body != {"status": "ok"}:
        raise ValidationError(f"Unexpected GET / response: {body}")


def _test_ask(client: TestClient) -> None:
    payload = {"question": "test question"}
    response = client.post("/ask", json=payload)
    if response.status_code != 200:
        raise ValidationError(f"POST /ask failed: {response.status_code}")

    body = response.json()
    if body.get("question") != payload["question"]:
        raise ValidationError(f"Unexpected question echo: {body}")

    answer = body.get("answer")
    if not isinstance(answer, list) or len(answer) != 2:
        raise ValidationError(f"Expected answer tuple serialized as list[2], got: {answer}")

    if answer[0] != "fake-answer:test question" or answer[1] != "fake-context":
        raise ValidationError(f"Unexpected answer content: {answer}")


def _test_rebuild_unauthorized(client: TestClient) -> None:
    response = client.post("/rebuild")
    if response.status_code != 401:
        raise ValidationError(f"POST /rebuild without auth should be 401, got {response.status_code}")


def _test_rebuild_authorized_success(client: TestClient, api_module) -> None:
    original_run = api_module.subprocess.run

    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    api_module.subprocess.run = fake_run
    try:
        response = client.post("/rebuild", auth=("admin", "password"))
    finally:
        api_module.subprocess.run = original_run

    if response.status_code != 200:
        raise ValidationError(f"POST /rebuild with auth failed: {response.status_code}")

    body = response.json()
    if body.get("status") != "ok":
        raise ValidationError(f"Unexpected rebuild response: {body}")


def main() -> int:
    try:
        api_module = _load_api_module()
        client = TestClient(api_module.app)

        _test_health(client)
        _test_ask(client)
        _test_rebuild_unauthorized(client)
        _test_rebuild_authorized_success(client, api_module)

        print("API endpoint tests passed")
        return 0
    except ValidationError as err:
        print(f"API endpoint tests failed: {err}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
