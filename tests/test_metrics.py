import threading
import time

from config import Target
from metrics import MetricsProbe, parse_gauge


def test_never_ok_means_not_down():
    n = [0.0]
    p = MetricsProbe(lambda u: "boom", lambda: n[0], authoritative=True)
    # no successful poll ever: misconfig, not death
    assert p.engine_down("http://x.test/metrics") is False


def test_ok_then_dead_and_authoritative_means_down():
    n = [100.0]
    p = MetricsProbe(lambda u: "", lambda: n[0], authoritative=True, poll_interval=10.0)
    p.tick("http://x.test/metrics", 0.4)
    n[0] = 140.0                                   # 40 s > 3 * 10 s
    assert p.engine_down("http://x.test/metrics") is True


def test_dead_but_not_authoritative_means_not_down():
    n = [100.0]
    p = MetricsProbe(lambda u: "", lambda: n[0], authoritative=False, poll_interval=10.0)
    p.tick("http://x.test/metrics", 0.4)
    n[0] = 140.0
    assert p.engine_down("http://x.test/metrics") is False


def test_parses_gauge_line():
    body = ('# HELP vllm:gpu_cache_usage_perc ...\n'
            'vllm:gpu_cache_usage_perc{api_server="0"} 0.888\n')
    assert parse_gauge(body, "vllm:gpu_cache_usage_perc") == 0.888
    assert parse_gauge("nothing", "vllm:gpu_cache_usage_perc") is None


def test_stop_interrupts_thread_within_one_interval():
    started = threading.Event()

    def fetch(u):
        started.set()
        return ""

    p = MetricsProbe(fetch, lambda: time.time(), False, poll_interval=0.05)
    p.start({"a": Target("a", 100, 100, ["http://a.test:1"], "http://a.test:1/metrics")})
    assert started.wait(5.0)
    t0 = time.time()
    p.stop()
    assert time.time() - t0 < 1.0
