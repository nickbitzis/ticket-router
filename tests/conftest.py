from types import SimpleNamespace

import pytest

from ticketrouter import data
from ticketrouter.evaluate import build_bench, train_classical


@pytest.fixture(scope="session")
def bench():
    """A small end-to-end bench: 3,000 tickets, simulated LLM tiers."""
    return build_bench(n=3000, seed=7)


@pytest.fixture(scope="session")
def drift_setup():
    tickets = data.generate(n=6000, seed=3)
    train, _, pool = data.partition(tickets, seed=3)
    model = train_classical(train)
    stream = data.drift_stream(pool, per_week=250, launch_week=7)
    return model, stream, train


class FakeMessages:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        r = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(r, Exception):
            raise r
        text, stop = (r, "end_turn") if isinstance(r, str) else r
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(input_tokens=420, output_tokens=60),
            stop_reason=stop,
            model=kw["model"],
        )


class FakeClient:
    def __init__(self, *replies):
        self.messages = FakeMessages(replies)


@pytest.fixture
def fake_client():
    return FakeClient
