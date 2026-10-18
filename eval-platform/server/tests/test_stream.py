"""StreamHub 测试：订阅、发布、慢消费者 lagged 与主题回收。"""

import asyncio

from eval_server.domain import RunStarted
from eval_server.stream import SUBSCRIBER_BUFFER, StreamHub, Topic


def _event(seq: int) -> RunStarted:
    return RunStarted(seq=seq, run_id="run-1", case_id="c1", task="t", model="m")


async def test_publish_reaches_subscribers() -> None:
    hub = StreamHub()
    topic = Topic.run("run-1")
    sub1 = hub.subscribe(topic)
    sub2 = hub.subscribe(topic)

    hub.publish(topic, _event(0))

    kind1, ev1 = await sub1.next()
    kind2, ev2 = await sub2.next()
    assert (kind1, kind2) == ("event", "event")
    assert ev1.seq == ev2.seq == 0


async def test_publish_without_subscribers_is_noop() -> None:
    hub = StreamHub()
    hub.publish(Topic.run("nobody"), _event(0))


async def test_topics_are_isolated() -> None:
    hub = StreamHub()
    sub_run = hub.subscribe(Topic.run("run-1"))
    sub_batch = hub.subscribe(Topic.batch("batch-1"))

    hub.publish(Topic.run("run-1"), _event(0))

    kind, ev = await sub_run.next()
    assert (kind, ev.seq) == ("event", 0)

    hub.publish(Topic.batch("batch-1"), _event(1))
    kind_b, ev_b = await sub_batch.next()
    assert (kind_b, ev_b.seq) == ("event", 1)


async def test_slow_subscriber_gets_lagged_signal() -> None:
    hub = StreamHub()
    topic = Topic.run("run-1")
    sub = hub.subscribe(topic)

    for i in range(SUBSCRIBER_BUFFER + 5):
        hub.publish(topic, _event(i))

    assert sub.lagged == 5
    kind, dropped = await sub.next()
    assert kind == "lagged"
    assert dropped == 5

    # lagged 信号之后继续读取缓冲区内的事件
    kind2, ev = await sub.next()
    assert kind2 == "event"
    assert 0 <= ev.seq < SUBSCRIBER_BUFFER


async def test_close_unsubscribes_but_buffer_remains() -> None:
    hub = StreamHub()
    topic = Topic.run("run-1")
    sub = hub.subscribe(topic)
    hub.publish(topic, _event(0))
    assert hub.subscriber_count(topic) == 1

    sub.close()
    assert hub.subscriber_count(topic) == 0
    sub.close()  # 幂等

    kind, ev = await sub.next()
    assert (kind, ev.seq) == ("event", 0)


async def test_retire_drops_topic() -> None:
    hub = StreamHub()
    topic = Topic.run("run-1")
    sub = hub.subscribe(topic)
    assert hub.subscriber_count(topic) == 1

    hub.retire(topic)
    assert hub.subscriber_count(topic) == 0
    hub.retire(topic)  # 幂等

    # retire 后发布不再投递给已注册的订阅（该订阅已被注销）
    hub.publish(topic, _event(1))
    with_notif = asyncio.ensure_future(sub.next())
    await asyncio.sleep(0.01)
    assert not with_notif.done()
    with_notif.cancel()


async def test_close_last_subscriber_removes_topic() -> None:
    hub = StreamHub()
    topic = Topic.batch("b1")
    sub = hub.subscribe(topic)
    hub.publish(topic, _event(0))
    sub.close()
    assert hub.subscriber_count(topic) == 0
