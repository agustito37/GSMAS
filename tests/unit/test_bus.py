"""Unit tests for the event bus.

Run with:
    uv run pytest
"""

import asyncio

import pytest

from core.events.bus import Event, EventBus


@pytest.mark.unit
def test_event_bus_subscribe():
    """subscribe registers the handler under the event type."""
    bus = EventBus()

    def handler(event):
        pass

    bus.subscribe("node_created", handler)

    assert handler in bus._handlers[("node_created", None)]


@pytest.mark.unit
async def test_event_bus_subscribe_by_node_type():
    """A handler narrowed to a node_type only receives events of that type."""
    bus = EventBus()
    evidence, hypothesis = [], []
    bus.subscribe("node_created", lambda e: evidence.append(e), node_type="Evidence")
    bus.subscribe("node_created", lambda e: hypothesis.append(e), node_type="Hypothesis")

    bus.publish(Event(type="node_created", node_id="e1", node_type="Evidence"))
    await bus.aclose()

    assert len(evidence) == 1
    assert evidence[0].node_id == "e1"
    assert hypothesis == []  # the Hypothesis-narrowed handler did not fire


@pytest.mark.unit
async def test_event_bus_subscribe_any_node_type():
    """A handler with node_type=None receives every node_type for the event."""
    bus = EventBus()
    received = []
    bus.subscribe("node_created", lambda e: received.append(e.node_type))  # any

    bus.publish(Event(type="node_created", node_id="e1", node_type="Evidence"))
    bus.publish(Event(type="node_created", node_id="h1", node_type="Hypothesis"))
    await bus.aclose()

    assert sorted(received) == ["Evidence", "Hypothesis"]


@pytest.mark.unit
async def test_event_bus_no_double_delivery_for_typeless_event():
    """An 'any' subscriber receives a node_type-less event exactly once."""
    bus = EventBus()
    received = []
    bus.subscribe("edge_created", lambda e: received.append(e))  # node_type=None

    bus.publish(Event(type="edge_created"))  # node_type is None
    await bus.aclose()

    assert len(received) == 1  # not 2


@pytest.mark.unit
async def test_event_bus_publish():
    """publish delivers the event to a subscribed handler."""
    bus = EventBus()
    received = []
    bus.subscribe("node_created", lambda e: received.append(e))

    bus.publish(Event(type="node_created", node_id="n1"))
    await bus.aclose()

    assert len(received) == 1
    assert received[0].node_id == "n1"


@pytest.mark.unit
async def test_event_bus_publish_multiple_events():
    """Every published event of a type reaches the handler."""
    bus = EventBus()
    received = []
    bus.subscribe("node_created", lambda e: received.append(e.node_id))

    bus.publish(Event(type="node_created", node_id="1"))
    bus.publish(Event(type="node_created", node_id="2"))
    await bus.aclose()

    assert sorted(received) == ["1", "2"]


@pytest.mark.unit
async def test_event_bus_publish_multiple_subscribers():
    """Broadcast: every subscriber to a type receives the event."""
    bus = EventBus()
    a, b = [], []
    bus.subscribe("node_created", lambda e: a.append(e))
    bus.subscribe("node_created", lambda e: b.append(e))

    bus.publish(Event(type="node_created", node_id="n1"))
    await bus.aclose()

    assert len(a) == 1
    assert len(b) == 1


@pytest.mark.unit
async def test_event_bus_publish_multiple_subscribers_multiple_events():
    """Each subscriber receives every event."""
    bus = EventBus()
    a, b = [], []
    bus.subscribe("node_created", lambda e: a.append(e.node_id))
    bus.subscribe("node_created", lambda e: b.append(e.node_id))

    bus.publish(Event(type="node_created", node_id="1"))
    bus.publish(Event(type="node_created", node_id="2"))
    await bus.aclose()

    assert sorted(a) == ["1", "2"]
    assert sorted(b) == ["1", "2"]


@pytest.mark.unit
async def test_event_bus_safe_run_sync():
    """A sync handler (lambda) is invoked."""
    bus = EventBus()
    received = []
    bus.subscribe("x", lambda e: received.append(e))

    bus.publish(Event(type="x"))
    await bus.aclose()

    assert len(received) == 1


@pytest.mark.unit
async def test_event_bus_safe_run_async():
    """An async handler is awaited and invoked."""
    bus = EventBus()
    received = []

    async def handler(event):
        received.append(event)

    bus.subscribe("x", handler)
    bus.publish(Event(type="x"))
    await bus.aclose()

    assert len(received) == 1


@pytest.mark.unit
async def test_event_bus_safe_run_failing(caplog):
    """A failing handler is isolated: it does not break the other handlers, the
    error is logged, and aclose does not raise."""
    bus = EventBus()
    received = []

    def failing(event):
        raise ValueError("boom")

    bus.subscribe("x", failing)
    bus.subscribe("x", lambda e: received.append(e))

    bus.publish(Event(type="x"))
    await bus.aclose()  # must not raise

    assert len(received) == 1  # the other handler still ran
    assert "event handler failed" in caplog.text


@pytest.mark.unit
async def test_event_bus_aclose():
    """aclose waits for in-flight handlers (publish is fire-and-forget)."""
    bus = EventBus()
    done = []

    async def slow(event):
        await asyncio.sleep(0.01)
        done.append(event)

    bus.subscribe("x", slow)
    bus.publish(Event(type="x"))

    assert done == []  # publish only scheduled it; the handler has not run yet

    await bus.aclose()

    assert len(done) == 1  # aclose waited for it to finish


@pytest.mark.unit
async def test_unsubscribe_stops_delivery():
    """After unsubscribe, the handler no longer receives events of that key."""
    bus = EventBus()
    received = []

    def handler(event):
        received.append(event.node_id)

    bus.subscribe("node_created", handler, node_type="Case")
    bus.publish(Event(type="node_created", node_id="c1", node_type="Case"))
    bus.unsubscribe("node_created", handler, node_type="Case")
    bus.publish(Event(type="node_created", node_id="c2", node_type="Case"))
    await bus.aclose()

    assert received == ["c1"]  # only the event published before unsubscribe


@pytest.mark.unit
def test_unsubscribe_unknown_handler_is_noop():
    """Unsubscribing a handler that was never registered does not raise."""
    bus = EventBus()

    def handler(event):
        pass

    bus.unsubscribe("node_created", handler, node_type="Case")  # must not raise
    assert handler not in bus._handlers.get(("node_created", "Case"), [])
