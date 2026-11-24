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
| `pipeline.joblib` | Standard price model (log price, trained on prices clipped to €69,000 to €1,949,050) |
| `luxury_classifier.joblib` | Probability that a property is in the top 1% of prices (above €1,949,050) |
| `pipeline_luxury.joblib` | Price model for that top 1%, trained on 111 listings |
| `pipeline_lower.joblib`, `pipeline_upper.joblib` | 10th and 90th percentile quantile models of the standard segment |
| `luxury_threshold.json` | Routing threshold: at or above 0.9 the luxury model prices the property |
| `interval_calibration.json` | Conformal margin that widens the quantile range to 80% coverage |
| `categories.json` | The categories the models were trained on (cities, provinces, states) |
| `locations.json` | Where each training postcode and city is (median coordinates of its listings), and each city's usual postcode, cities filed by province |

## How to use

The pipelines read engineered features, so use them through the repository's `api/predict.py` rather than on raw input:

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

The pickles need `scikit-learn==1.9.0`, `xgboost==3.2.0` and `category_encoders==2.8.1`. `.joblib` files are pickles: only load them from a repository you trust.

## Evaluation

Test split of `src/train.py`: 2,361 listings that nothing was fitted, tuned or chosen on. Routing at 0.9, as deployed:

| Tier | MAPE |
|---|---:|
| All listings | 20.3% |
| < €1M | 19.8% |
| €1M to €1.95M | 27.4% |
| >= €1.95M (21 listings) | 43.9% |

Median error 12.9%. 95% bootstrap interval on the overall MAPE: 19.1% to 21.6%. The 10th to 90th percentile range held the real price on 80.4% of the 2,344 standard-segment test listings (target 80%; 70.8% before calibration on a separate split). It holds less often for estimates above €500k (about 74%), and the luxury segment gets no range.

## Training data and limitations

About 11,000 training listings scraped once from Immovlan: asking prices, not sale prices. The data is not redistributed.

- An estimation tool, not a valuation.
- Estimates run high under €150k (median +22%) and low between €500k and €1M (median -10%).
- The location is the centroid of the postcode's training listings (the city's if the postcode has none).
- Above €1.95M prices are underestimated (-41% bias with routing, -51% without).
- A left-out build year is priced at the training median (1974); a city outside the training vocabulary is ignored.
