---
library_name: xgboost
pipeline_tag: tabular-regression
tags:
  - tabular-regression
  - xgboost
  - scikit-learn
  - real-estate
  - belgium
---

# Immo Eliza: Belgian property price models

Five scikit-learn pipelines (target encoding + XGBoost) that estimate the asking
price of a Belgian house or apartment, as served by the API and the web app in
[github.com/Siegried81/immo-eliza-deployment](https://github.com/Siegried81/immo-eliza-deployment).

## Files

| File | Role |
|---|---|
| `pipeline.joblib` | Standard price model (log price, trained on prices clipped to €69,000 to €1,948,950) |
| `luxury_classifier.joblib` | Probability that a property is in the top 1% of prices (above €1,948,950) |
| `pipeline_luxury.joblib` | Price model for that top 1%, trained on 111 listings |
| `pipeline_lower.joblib`, `pipeline_upper.joblib` | 10th and 90th percentile quantile models of the standard segment |
| `luxury_threshold.json` | Routing threshold: at or above 0.9 the luxury model prices the property |
| `interval_calibration.json` | Conformal margin that widens the quantile range to 80% coverage |
| `categories.json` | The categories the models were trained on (cities, provinces, states) |

## How to use

The pipelines read engineered features (`src/features.py::add_features`) and
the training vocabulary (`to_training_vocabulary`), so use them through the
repository's `api/predict.py` rather than on raw input:

```bash
git clone https://github.com/Siegried81/immo-eliza-deployment && cd immo-eliza-deployment
pip install -r requirements.txt
hf download Siegried81/immo-eliza-price-models --local-dir models
```

```python
from api.predict import THRESHOLD_PATH, engine, load_routing_threshold, route

result = engine.predict({"postcode": 1050, "province": "Brussels", "city": "Ixelles",
                         "property_type": "APARTMENT", "livable_surface": 90})
print(route(result, load_routing_threshold(THRESHOLD_PATH)))
```

Loading needs the versions the models were pickled with:
`scikit-learn==1.9.0`, `xgboost==3.2.0`, `category_encoders==2.8.1`.
`.joblib` files are pickles: only load them from a repository you trust.

## Evaluation

All figures are on the test split of `src/train.py`: 2,362 listings that
nothing was fitted, early-stopped, tuned or chosen on. Routing at 0.9, as
deployed:

| Tier | MAPE |
|---|---:|
| All listings | 26.5% (22.5% without one bad row) |
| < €1M | 26.0% (21.8% without it) |
| €1M to €1.95M | 34.7% |
| ≥ €1.95M (21 listings) | 46.2% |

The bad row is a 77 m² apartment listed at €3,048, a data error: on its own
it adds 4 points to the MAPE. Median error: 15.0%. The standard model alone:
26.24% MAPE (22.17% without that row), MAE €92,024, R² 0.646 (euros).

The 10th to 90th percentile range held the real price 80.9% of the time on
2,344 test listings (target 80%; 73.6% before calibration). Its margin was set
on a separate calibration split that no model early-stopped on.

Until 2026-11-10 this card quoted a validation split (23.0%, 77.9%) that also
served early stopping, tuning and the routing threshold, so those figures were
a bit optimistic.

## Training data and limitations

About 11,000 training listings, scraped once from Immovlan: asking prices, not sale
prices, at one point in time. The data itself is not redistributed here.

- This is an estimation tool, not a valuation.
- Below €1M prices are overestimated by about 7% on average.
- Above €1.95M prices are underestimated (−43% bias with routing, −52% without).
- A left-out build year is priced at the training median (1974).
- A city outside the training vocabulary is accepted but ignored.
