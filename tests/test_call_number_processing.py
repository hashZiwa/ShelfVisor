import unittest

from app.analysis.call_number_processing import call_number_sort_key, check_call_number_order


class CallNumberProcessingTests(unittest.TestCase):
    def test_numeric_segments_sort_numerically(self):
        values = ["811.20 B", "811.3 A"]
        self.assertEqual(sorted(values, key=call_number_sort_key), ["811.3 A", "811.20 B"])

    def test_current_status_behavior_is_preserved(self):
        result = check_call_number_order(["A", "B"])
        self.assertEqual([item.status for item in result], ["ok", "ok"])


if __name__ == "__main__":
    unittest.main()
