# Historical notebook

`hotel_cancellation_modeling.ipynb` is preserved as the original exploratory project artifact.

Because the large notebook and its modified source dataset cannot be reliably re-executed and audited end-to-end through the current repository connection, it is not presented as the canonical implementation.

Use [`../src/hotel_cancellation_model.py`](../src/hotel_cancellation_model.py) for the audited baseline. It places imputation, scaling and categorical encoding inside cross-validation pipelines, uses stratified held-out evaluation and does not require machine-specific data paths.
