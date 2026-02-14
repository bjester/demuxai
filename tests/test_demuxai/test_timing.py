import time
from unittest import TestCase

from demuxai.timing import Timing
from demuxai.timing import TimingStatistics


class TimingTestCase(TestCase):
    def test_init(self):
        timing = Timing()
        self.assertIsNone(timing.start_time)
        self.assertIsNone(timing.first_byte_time)
        self.assertIsNone(timing.end_time)

    def test_start_set_first_byte_end(self):
        timing = Timing()
        timing.start()
        time.sleep(0.01)  # Simulate some time passing
        timing.set_first_byte_received()
        time.sleep(0.01)  # Simulate some more time passing
        timing.end()

        self.assertIsNotNone(timing.start_time)
        self.assertIsNotNone(timing.first_byte_time)
        self.assertIsNotNone(timing.end_time)
        self.assertGreater(timing.first_byte_time, timing.start_time)
        self.assertGreater(timing.end_time, timing.first_byte_time)

        self.assertGreater(timing.time_to_first_byte, 0)
        self.assertGreater(timing.duration, 0)
        self.assertGreater(timing.response_duration, 0)
        self.assertAlmostEqual(
            timing.duration,
            timing.time_to_first_byte + timing.response_duration,
            delta=0.001,
        )

    def test_properties_before_set(self):
        timing = Timing()
        with self.assertRaises(AssertionError, msg="First byte not received"):
            _ = timing.time_to_first_byte
        with self.assertRaises(AssertionError, msg="Timing not ended"):
            _ = timing.duration
        with self.assertRaises(AssertionError, msg="Timing not ended"):
            _ = timing.response_duration

        timing.start()
        with self.assertRaises(AssertionError, msg="First byte not received"):
            _ = timing.time_to_first_byte
        with self.assertRaises(AssertionError, msg="Timing not ended"):
            _ = timing.duration
        with self.assertRaises(AssertionError, msg="First byte not received"):
            _ = timing.response_duration

        timing.set_first_byte_received()
        with self.assertRaises(AssertionError, msg="Timing not ended"):
            _ = timing.duration
        with self.assertRaises(AssertionError, msg="Timing not ended"):
            _ = timing.response_duration

    def test_context_manager(self):
        with Timing() as timing:
            time.sleep(0.01)
            timing.set_first_byte_received()
            time.sleep(0.01)

        self.assertIsNotNone(timing.start_time)
        self.assertIsNotNone(timing.first_byte_time)
        self.assertIsNotNone(timing.end_time)
        self.assertGreater(timing.time_to_first_byte, 0)
        self.assertGreater(timing.duration, 0)
        self.assertGreater(timing.response_duration, 0)

    def test_str(self):
        with Timing() as timing:
            time.sleep(0.01)
            timing.set_first_byte_received()
            time.sleep(0.01)
        self.assertIsInstance(str(timing), str)
        self.assertIn("s /", str(timing))


class TimingStatisticsTestCase(TestCase):
    def test_init(self):
        stats = TimingStatistics()
        self.assertEqual(stats.limit, 100)
        self.assertEqual(stats.timings, [])

        stats = TimingStatistics(limit=5)
        self.assertEqual(stats.limit, 5)

    def test_add(self):
        stats = TimingStatistics(limit=2)
        timing1 = Timing()
        timing1.start()
        timing1.set_first_byte_received()
        timing1.end()
        stats.add(timing1)
        self.assertEqual(len(stats.timings), 1)
        self.assertEqual(stats.timings[0], timing1)

        timing2 = Timing()
        timing2.start()
        timing2.set_first_byte_received()
        timing2.end()
        stats.add(timing2)
        self.assertEqual(len(stats.timings), 2)
        self.assertEqual(stats.timings[1], timing2)

        timing3 = Timing()
        timing3.start()
        timing3.set_first_byte_received()
        timing3.end()
        stats.add(timing3)
        self.assertEqual(len(stats.timings), 2)
        self.assertEqual(stats.timings[0], timing2)  # Oldest should be popped
        self.assertEqual(stats.timings[1], timing3)

    def test_properties(self):
        stats = TimingStatistics(limit=10)
        self.assertEqual(stats.time_to_first_byte, 0)
        self.assertEqual(stats.duration, 0)
        self.assertEqual(stats.response_duration, 0)

        timing1 = Timing()
        timing1.start_time = 0
        timing1.first_byte_time = 0.1
        timing1.end_time = 0.3
        stats.add(timing1)

        timing2 = Timing()
        timing2.start_time = 0
        timing2.first_byte_time = 0.2
        timing2.end_time = 0.5
        stats.add(timing2)

        # Expected averages
        expected_ttfb = (0.1 + 0.2) / 2
        expected_duration = (0.3 + 0.5) / 2
        expected_response_duration = ((0.3 - 0.1) + (0.5 - 0.2)) / 2

        self.assertAlmostEqual(stats.time_to_first_byte, expected_ttfb)
        self.assertAlmostEqual(stats.duration, expected_duration)
        self.assertAlmostEqual(stats.response_duration, expected_response_duration)

    def test_str(self):
        stats = TimingStatistics(limit=10)
        self.assertEqual(str(stats), "0.000s / 0.000s / 0.000s")

        timing1 = Timing()
        timing1.start_time = 0
        timing1.first_byte_time = 0.123
        timing1.end_time = 0.456
        stats.add(timing1)

        self.assertEqual(str(stats), "0.123s / 0.333s / 0.456s")
