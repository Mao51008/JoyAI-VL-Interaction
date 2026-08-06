import unittest

from training.omni.projector_stage1.progress import progress_iter


class FakeProgress:
    def __init__(self, iterable, **kwargs):
        self.iterable = iterable
        self.kwargs = kwargs

    def __iter__(self):
        return iter(self.iterable)


class ProgressTest(unittest.TestCase):
    def test_progress_has_shard_label_and_actual_total(self):
        captured = {}

        def factory(iterable, **kwargs):
            captured.update(kwargs)
            return FakeProgress(iterable, **kwargs)

        values = list(progress_iter(range(3), total=3, description="shard 2/4",
                                     unit="sample", no_progress=False, tqdm_factory=factory))
        self.assertEqual(values, [0, 1, 2])
        self.assertEqual(captured["total"], 3)
        self.assertEqual(captured["desc"], "shard 2/4")
        self.assertEqual(captured["unit"], "sample")
        self.assertEqual(captured["mininterval"], 1.0)

    def test_no_progress_returns_original_iterable(self):
        values = [1, 2]
        called = False

        def factory(*_args, **_kwargs):
            nonlocal called
            called = True
            return None

        result = progress_iter(values, total=len(values), description="cache features",
                               unit="sample", no_progress=True, tqdm_factory=factory)
        self.assertIs(result, values)
        self.assertFalse(called)


if __name__ == "__main__":
    unittest.main()
