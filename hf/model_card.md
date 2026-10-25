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

# Immo Eliza — Belgian property price models

Five scikit-learn pipelines (target encoding + XGBoost) that estimate the asking
price of a Belgian house or apartment, as served by the API and the web app in
[github.com/Siegried81/immo-eliza-deployment](https://github.com/Siegried81/immo-eliza-deployment).

## Files

| File | Role |
|---|---|
| `pipeline.joblib` | Standard price model (log price, trained on prices clipped to €68,475–€1,890,000) |
| `luxury_classifier.joblib` | Probability that a property is in the top 1% of prices (≥ €1,890,000) |
| `pipeline_luxury.joblib` | Price model for that top 1%, trained on 125 listings |
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

All figures are on 3,150 listings that no model was fitted on (the validation
split of `src/train.py`), with routing at 0.9 as deployed:

| Tier | MAPE |
|---|---:|
| All listings | 23.0% |
| < €1M | 22.4% |
| €1M–€1.89M | 35.3% |
| ≥ €1.89M (36 listings) | 35.8% |

The 10th–90th percentile range held the real price 77.9% of the time on 1,555
listings not used to calibrate it (target 80%; 70.7% before calibration).

## Training data and limitations

About 12,600 listings scraped once from Immovlan: asking prices, not sale
prices, at one point in time. The data itself is not redistributed here.

- This is an estimation tool, not a valuation.
- Below €1M prices are overestimated by about 7% on average.
- Above €1.89M prices are underestimated (−29% bias with routing, −45% without).
- A city outside the training vocabulary is accepted but ignored.
