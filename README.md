# Hotel Cancellation Prediction

An applied machine-learning project focused on identifying the factors associated with hotel booking cancellations and translating the analysis into practical revenue-protection recommendations.

## Business problem

Hotel cancellations create avoidable revenue uncertainty. This project uses historical reservation data to investigate which booking characteristics are most strongly associated with cancellation and how those patterns could inform operational decisions.

## Repository structure

- [`hotel_cancellation_modeling.ipynb`](hotel_cancellation_modeling.ipynb) — primary analysis notebook.
- [`.gitignore`](.gitignore) — excludes local notebook, Python and R environment artifacts.

## Dataset

The analysis works with hotel reservation features including:

- number of adults and children,
- weekend and weekday nights,
- meal plan,
- parking requirements,
- room type,
- lead time,
- arrival date,
- market segment,
- repeat-guest status,
- previous cancellations and completed bookings,
- average room price,
- special requests,
- final booking status.

The original source is the Hotel Reservations Classification dataset on Kaggle; the project notes that the exercise data were modified.

## Analytical objective

The goal is to identify the variables that contribute most to whether a booking is fulfilled or cancelled and connect those findings to practical actions that could reduce cancellation exposure.

## Reproducing the analysis

1. Open `hotel_cancellation_modeling.ipynb` in a compatible Jupyter environment.
2. Install the R/Python kernel and packages referenced by the notebook.
3. Obtain the source or modified project dataset used by the analysis.
4. Update the local data path if required and run the notebook sequentially.

## Scope

This is an applied portfolio project demonstrating how machine learning can be connected to a concrete hospitality business problem. It is not a deployed cancellation-scoring service.
