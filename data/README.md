# Data

Place the hotel-reservation dataset used by the project here as:

```text
data/hotel_reservations.csv
```

The historical project notes identify the exercise as based on a **Hotel Reservations Classification** dataset and also state that the exercise data were modified. The exact modified CSV, transformation recipe, source version and checksum are not present in the repository.

For that reason, this repository does **not** claim that the original historical dataset can be reconstructed by downloading an arbitrary similarly named public dataset. The audited script is intentionally schema-flexible: it separates numeric/categorical predictors automatically and attempts to infer common booking-status target names.

If you have the exact project CSV, place it at the path above and record its provenance/checksum. If the target cannot be inferred, pass the exact column name explicitly:

```bash
python src/hotel_cancellation_model.py --target Booking_Status
```

You can also point the workflow to another compatible CSV with `--data`, but results from a different dataset should be reported as a new analysis rather than as a reproduction of the historical exercise.
