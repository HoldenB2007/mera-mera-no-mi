# Mera Mera no Mi — Colorado wildfire risk prediction

Predicts the daily probability of wildfire activity for every 4 km × 4 km grid cell in Colorado from weather, terrain, and fire-history data, and shows the result on an interactive map. Built by two students to help a local fire department think about where risk is concentrating — it is a research prototype, not an operational tool.

**Authors:** [Everett Rike](https://github.com/everettrike) (data pipeline, Random Forest baseline, Streamlit map) and [Holden Bronson](https://github.com/HoldenB2007) (PyTorch classifier, data prep, evaluation).

---

## What it does

1. **Builds a 20-year dataset (2006–2026)** for Colorado with Google Earth Engine, one row per grid cell per day:
   - **Weather (GRIDMET, daily):** precipitation, max temperature, min relative humidity, wind speed, and Energy Release Component (a standard fire-danger index).
   - **Terrain (SRTM, static):** elevation, slope, aspect.
   - **Fire (FIRMS):** MODIS brightness temperature (`T21`) — any detection in a cell that day is labeled `fire = 1`.
   - Each variable is reduced to max / mean / std-dev over the cell, so mountainous cells are distinguishable from flat ones.
2. **Trains two classifiers** on a temporal split — train on 2006–2020, test on 2021–2026 — so the evaluation is a true forecast, not a random shuffle.
3. **Serves a Streamlit map** (`app.py`) where you pick a model and a date and see predicted fire risk per cell over an OpenTopoMap base layer, with actual detections marked.

## Results

Fire is rare (~8% of test rows even after undersampling no-fire days), so accuracy is misleading; the fire-class metrics are what matter.

| Model | Fire precision | Fire recall | Fire F1 | Notes |
|---|---|---|---|---|
| Random Forest (100 trees, `class_weight='balanced'`, threshold 0.6) | 0.47 | 0.02 | 0.04 | Almost never predicts fire at this threshold |
| **FireNet** (3-layer MLP, PyTorch, Adam, class-weighted BCE, threshold 0.6) | 0.36 | 0.37 | **0.37** | Recall up ~18× over the baseline |

Test set: 45,521 cell-days from 2021 onward (3,589 with fire). Both models use identical features, splits, and threshold.

**Honest read:** FireNet is a large improvement in recall over the baseline, but a 0.37 F1 is a prototype-grade result. Part of the RF baseline's weakness is the fixed 0.6 threshold on a class-weighted model — see *Next steps*.

## Repository layout

| File | Purpose |
|---|---|
| `FireData.ipynb` | Earth Engine export: filters GRIDMET/FIRMS/SRTM to Colorado, builds the 4 km UTM-13N grid, joins daily weather + fire + terrain, exports one CSV per year to Google Drive |
| `RandomForestModel.ipynb` | Loads the CSVs, undersamples no-fire days, temporal split, Random Forest baseline, saves `fire_prediction_model.pkl` |
| `PyTorchAdamModel.ipynb` | Same data prep; standardization; `FireNet` MLP with dropout and a 9.46× positive-class weight; saves `pytorch_fire_model.pth` + `pytorch_scaler.pkl` |
| `DemoData.ipynb` | Prepares the subset of dates/cells the demo map displays |
| `app.py` | Streamlit + Folium demo: model selector, date picker, per-cell risk rectangles, fire markers |
| `requirements.txt` | pandas, scikit-learn, torch, earthengine-api, geemap, streamlit, streamlit-folium, folium, joblib |

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Data (only if regenerating; ~20 min of Earth Engine export tasks)
#    Open FireData.ipynb, run ee.Authenticate(), run all cells.
#    CSVs land in Google Drive under EarthEngineData/; copy them to ./FireData/

# 2. Train (each notebook expects CSVs in ./FireData/ — edit `path` at the top)
#    Run RandomForestModel.ipynb and/or PyTorchAdamModel.ipynb end to end.

# 3. Demo
streamlit run app.py
```

You need a Google Earth Engine account for step 1. Steps 2–3 work from the exported CSVs alone.

## Design notes

- **Why a 4 km grid in UTM zone 13N:** matches GRIDMET's native resolution and keeps cell areas equal across the state; Colorado sits entirely in zone 13N.
- **Why undersample no-fire days:** at the native ratio (~0.1% fire), both models collapse to "never fire." Sampling ~0.13% of no-fire rows per year gives ~10% positives while keeping every fire day.
- **Why a temporal split:** a random split would leak weather regimes across train/test and inflate every metric. Forecasting the future from the past is the honest test.
- **Why class-weighted loss:** mirrors the RF's `class_weight='balanced'` so the two models are compared on architecture, not on imbalance handling.

## Known limitations

- Labels come from satellite detections, so cloud cover and small fires are under-labeled.
- Features are same-day; there is no lag/lead structure (e.g., yesterday's ERC predicting today's ignition).
- The demo loads pre-selected dates, not live data.
- No spatial cross-validation — nearby cells on the same day are highly correlated.

## Next steps

1. Sweep the decision threshold and report **PR-AUC** for both models instead of F1 at a fixed 0.6 — the RF baseline is likely being held back by the cutoff more than by the model.
2. Add lagged weather features (1-, 3-, 7-day ERC and humidity) and day-of-season.
3. Try gradient-boosted trees (LightGBM/XGBoost), which usually win on tabular data of this shape.
4. Spatial block cross-validation to get an honest variance estimate.
5. Pull live GRIDMET for "today" so the map can show current risk instead of historical dates.

## Data sources

- [GRIDMET](https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_GRIDMET) — University of Idaho gridded surface meteorology
- [FIRMS](https://developers.google.com/earth-engine/datasets/catalog/FIRMS) — NASA Fire Information for Resource Management System
- [SRTM GL1](https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003) — 30 m elevation
- [TIGER 2018 states](https://developers.google.com/earth-engine/datasets/catalog/TIGER_2018_States) — Colorado boundary
