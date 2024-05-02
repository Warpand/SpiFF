from unittest import TestCase

from spiff.cfg import Config


class TestConfig(TestCase):
    class DummyInnerConfig(Config):
        def __init__(self):
            self.a = 1.0
            self.b = 1.0

    class DummyConfig(Config):
        def __init__(self):
            self.x = 1
            self.y = "2"
            self.z = [1, 2, 3]
            self.inner = TestConfig.DummyInnerConfig()

    def test_override_when_ok_with_no_inner(self):
        cfg = TestConfig.DummyConfig()
        overrides = {"x": 2, "y": "1"}
        cfg.override(overrides)

        self.assertEqual(2, cfg.x)
        self.assertEqual("1", cfg.y)
        self.assertListEqual([1, 2, 3], cfg.z)
        self.assertEqual(1.0, cfg.inner.a)
        self.assertEqual(1.0, cfg.inner.b)

    def test_override_ok_with_inner(self):
        cfg = TestConfig.DummyConfig()
        overrides = {"z": [2, 3, 4], "inner": {"a": 2.0}}
        cfg.override(overrides)

        self.assertEqual(1, cfg.x)
        self.assertEqual("2", cfg.y)
        self.assertListEqual([2, 3, 4], cfg.z)
        self.assertEqual(2.0, cfg.inner.a)
        self.assertEqual(1.0, cfg.inner.b)

    def test_override_with_error(self):
        cfg = TestConfig.DummyConfig()
        overrides = {"x": 2, "WRONG": 0}

        with self.assertRaises(ValueError) as cm:
            cfg.override(overrides)
        msg = str(cm.exception)
        self.assertTrue("DummyConfig" in msg and "WRONG" in msg)

    def test_override_with_error_in_inner(self):
        cfg = TestConfig.DummyConfig()
        overrides = {"x": 2, "inner": {"a": 2.0, "WRONG": 0}}

        with self.assertRaises(ValueError) as cm:
            cfg.override(overrides)
        msg = str(cm.exception)
        self.assertTrue("DummyInnerConfig" in msg and "WRONG" in msg)

    def test_weird_case(self):
        cfg = TestConfig.DummyConfig()
        overrides = {"override": 0}
        self.assertRaises(ValueError, lambda: cfg.override(overrides))
