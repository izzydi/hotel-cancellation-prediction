# Predicting and Reducing Hotel Booking Cancellations

An applied machine-learning project focused on identifying the factors associated with hotel booking cancellations and translating the analysis into practical revenue-protection recommendations.

## Business problem

Hotel cancellations create avoidable revenue uncertainty. This project uses historical reservation data to investigate which booking characteristics are most strongly associated with cancellation and how those patterns could inform operational decisions.

## Repository contents

- [`hotel_cancellations_R_v07.ipynb`](hotel_cancellations_R_v07.ipynb) — complete analysis notebook.

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

The original data source is the Hotel Reservations Classification dataset on Kaggle; the project notes that the data used for the exercise were modified.

## Analytical objective

The goal is to identify the variables that contribute most to whether a booking is fulfilled or cancelled, then use those findings to support recommendations aimed at reducing cancellation risk.

## Reproducing the analysis

1. Open `hotel_cancellations_R_v07.ipynb` in a compatible Jupyter environment.
2. Install the R/Python kernel and packages referenced by the notebook as required.
3. Obtain the source dataset or the modified project dataset used by the notebook.
4. Update any local data path and run the notebook sequentially.

## Scope

This repository is an applied portfolio project demonstrating how machine learning can be connected to a concrete hospitality business problem. It is not a deployed cancellation-scoring service.
