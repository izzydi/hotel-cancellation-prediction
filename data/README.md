# Data

Place the hotel-reservation dataset used by the project here as:

```text
data/hotel_reservations.csv
```

The original project was based on the Hotel Reservations Classification dataset and notes that the exercise data were modified. The audited script therefore does not hard-code a full schema: it automatically separates numeric and categorical predictors and attempts to infer common booking-status target names.

If the target cannot be inferred, run the script with an explicit column name:

```bash
python src/hotel_cancellation_model.py --target Booking_Status
```

The project dataset is not committed to this repository.
