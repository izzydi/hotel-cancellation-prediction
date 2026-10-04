import unittest

from src.hotel_cancellation_model import infer_target, normalize_name


class HotelPipelineSmokeTests(unittest.TestCase):
    def test_normalize_name(self):
        self.assertEqual(normalize_name("Booking_Status"), "bookingstatus")

    def test_infer_common_target(self):
        columns = ["lead_time", "Booking_Status"]
        self.assertEqual(infer_target(columns, None), "Booking_Status")

    def test_requested_target_is_respected(self):
        columns = ["lead_time", "cancelled"]
        self.assertEqual(infer_target(columns, "cancelled"), "cancelled")


if __name__ == "__main__":
    unittest.main()
