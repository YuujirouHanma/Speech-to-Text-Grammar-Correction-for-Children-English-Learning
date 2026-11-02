"""BodySizeLimitMiddleware: the 413 must reach clients that are still sending."""

import asyncio

from starlette.types import Message, Receive, Scope, Send

from speakfix.api.middleware import BodySizeLimitMiddleware


def run(declared: int, messages: list[Message]) -> list[str]:
    events: list[str] = []
    pending = list(messages)

    async def receive() -> Message:
        events.append("receive")
        return pending.pop(0)

    async def send(message: Message) -> None:
        events.append(message["type"])

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        raise AssertionError("an oversized request must not reach the app")

    scope: Scope = {
        "type": "http",
        "headers": [(b"content-length", str(declared).encode())],
        "state": {},
    }
    asyncio.run(BodySizeLimitMiddleware(app, max_bytes=100)(scope, receive, send))
    return events


def test_oversized_body_is_read_before_answering() -> None:
    events = run(
        300,
        [
            {"type": "http.request", "body": b"x" * 200, "more_body": True},
            {"type": "http.request", "body": b"x" * 100, "more_body": False},
        ],
    )
    assert events == ["receive", "receive", "http.response.start", "http.response.body"]


def test_client_disconnect_stops_the_draining() -> None:
    events = run(300, [{"type": "http.disconnect"}])
    assert events == ["receive", "http.response.start", "http.response.body"]


def test_far_oversized_body_is_not_waited_for() -> None:
    assert run(10_000, []) == ["http.response.start", "http.response.body"]
