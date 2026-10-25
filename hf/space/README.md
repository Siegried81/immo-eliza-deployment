---
title: Immo Eliza
emoji: 🏠
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
short_description: Price estimates for Belgian houses and apartments
---

# Immo Eliza — Belgian property price estimates

Enter a house or an apartment and get a price estimate with its likely range
(10th–90th percentile), the price per m² and the segment that priced it.

- **Models:** XGBoost pipelines trained on ~12,600 listings scraped from
  Immovlan. A routing classifier sends likely luxury properties (≈ €1.9M and
  above) to a dedicated luxury model. The models are published separately at
  [Siegried81/immo-eliza-price-models](https://huggingface.co/Siegried81/immo-eliza-price-models).
- **Accuracy:** about 23% average error (23.0% MAPE) on 3,150 listings no model
  was fitted on; 22.4% below €1M. The range is calibrated to hold the real price
  8 times in 10; it held it 77.9% of the time on listings not used to set it.
- **This is an estimation tool, not a valuation.** Properties above €1.89M are
  typically underestimated.

This Space runs the FastAPI service and the Streamlit UI in one container.
Source code, evaluation and known limitations:
[github.com/Siegried81/immo-eliza-deployment](https://github.com/Siegried81/immo-eliza-deployment).
