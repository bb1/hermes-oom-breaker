from state import Store


class DictSink:
    def __init__(self):
        self.d = {}

    def get(self, k, d=None):
        return self.d.get(k, d)

    def set(self, k, v):
        self.d[k] = v


def test_set_get_roundtrip():
    s = Store(DictSink())
    s.set("L", {"a": 245760})
    assert s.get("L") == {"a": 245760}


def test_update_transforms_atomically():
    s = Store(DictSink())
    s.update(lambda d: {**d, "hits": 1})
    assert s.update(lambda d: {**d, "hits": d.get("hits", 0) + 1})["hits"] == 2


def test_corrupt_read_fails_open():
    class Boom:
        def get(self, k, d=None):
            raise OSError("corrupt")

        def set(self, k, v):
            pass

    s = Store(Boom())
    assert s.get("x", 5) == 5 and s.last_error
