# Hotel Cancellation Prediction

An applied machine-learning project focused on identifying booking characteristics associated with hotel cancellations and connecting those patterns to practical revenue-protection decisions.

## Primary workflow

[`src/hotel_cancellation_model.py`](src/hotel_cancellation_model.py) is the audited implementation. It creates a stratified hold-out set, automatically separates numeric and categorical predictors, and places imputation, one-hot encoding and scaling inside scikit-learn pipelines so learned preprocessing remains inside cross-validation folds.

The script tunes a balanced logistic-regression baseline and a balanced Random Forest on training data only, then reports accuracy, balanced accuracy and weighted F1 on the untouched test set. Metrics are written to `outputs/metrics.json`.

## Repository structure

- [`src/hotel_cancellation_model.py`](src/hotel_cancellation_model.py) — audited modelling pipeline.
- [`data/README.md`](data/README.md) — expected dataset layout and target handling.
- [`requirements.txt`](requirements.txt) — direct Python dependencies.
- [`archive/hotel_cancellation_modeling.ipynb`](archive/hotel_cancellation_modeling.ipynb) — original exploratory notebook retained for provenance.
- [`.gitignore`](.gitignore) — local environment/data exclusions.

## Dataset

The original project was based on the Hotel Reservations Classification dataset and notes that the exercise data were modified. The dataset includes adults/children, weekday and weekend stays, meal plan, parking, room type, lead time, arrival information, market segment, repeat-guest history, previous cancellations/completed bookings, room price, special requests and booking status.

Because the modified project dataset is not committed, the audited script accepts an explicit data path/target when needed.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/hotel_cancellation_model.py
```

If the booking-status column cannot be inferred automatically:

```bash
python src/hotel_cancellation_model.py --target Booking_Status
```

## Scope

This is an applied portfolio project demonstrating reproducible classification around a hospitality business problem. It is not a deployed cancellation-scoring service.
