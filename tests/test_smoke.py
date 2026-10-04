import unittest

from src.hotel_cancellation_model import (
    find_identifier_columns,
    infer_target,
    normalize_name,
)


class HotelPipelineSmokeTests(unittest.TestCase):
    def test_normalize_name(self):
        self.assertEqual(normalize_name("Booking_Status"), "bookingstatus")

    def test_infer_common_target(self):
        columns = ["lead_time", "Booking_Status"]
        self.assertEqual(infer_target(columns, None), "Booking_Status")

    def test_requested_target_is_respected(self):
        columns = ["lead_time", "cancelled"]
        self.assertEqual(infer_target(columns, "cancelled"), "cancelled")

    def test_common_booking_identifier_is_excluded(self):
        columns = ["Booking_ID", "lead_time", "room_type"]
        self.assertEqual(find_identifier_columns(columns), ["Booking_ID"])


if __name__ == "__main__":
    unittest.main()
