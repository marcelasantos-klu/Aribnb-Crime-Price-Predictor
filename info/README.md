## Quickstart (clone, set up, run)

Prerequisites: Python 3.9+ and Git installed locally. In VS Code, use the integrated terminal for the commands below and select the created virtual environment as your interpreter (Command Palette → “Python: Select Interpreter”).

1) Clone the repository  
```bash
git clone https://github.com/<your-account>/Dynamic-Pricing-in-Retail-Fashion-Industry-Data-Science.git
cd Dynamic-Pricing-in-Retail-Fashion-Industry-Data-Science
```

2) Create and activate a virtual environment (Python 3.9+ recommended)  
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

3) Install dependencies (required for the modelling scripts)  
```bash
python -m pip install -r info/requirements.txt
```

4) Run the full modelling pipeline (regression/classification/clustering)  
```bash
python "src/Step 4/final_models_switch.py"
```
- Outputs (plots, CSVs, models) are written under `plots&models/` with separate folders for with/without train-only outlier filtering.

5) (Optional) Run the exploratory maps/EDA from earlier steps  
Check the scripts under `src/Step 2/` for EDA and mapping utilities; they expect the cleaned data from `data/`.

## Project structure and module overview

- `data/`  
  Raw and intermediate datasets (merged Airbnb + crime in Step 1, cleaned in Step 3). Not tracked here.

- `src/Step 1/` (data acquisition & merge)  
  - `merge_airbnb.py`: merges Airbnb city files into a unified dataset.  
  - `merge_with_crime.py`: joins Airbnb data with crime indicators per city.  
  - `overviewBothDataSetFirst.py`: quick schema/column checks; `overview_output.txt` contains the summary.

- `src/Step 2/` (EDA & geography)  
  - `edaOnMergedDataSet.py`: main exploratory analysis (distributions, correlations, revenue buckets) and map generation.  
  - `geo_maps.py`: modularized map creation; saves PNG/HTML maps to `plots&models/EDA/maps`.  
  - `extract_geo_duplicates.py`: identifies/removes duplicate geo entries.  
  - `eda_terminal_output.txt`: captured EDA log.  
  Outputs: EDA plots and maps under `plots&models/EDA/`.

- `src/Step 3/` (data cleaning for modelling)  
  - `CleanUpBeforeModeling.py`: imputations, filtering, final feature selection; writes the cleaned dataset used in Step 4.

- `src/Step 4/` (modelling)  
  - `config.py`: run configuration (split strategy, seeds, toggles).  
  - `data_io.py`: loading utilities and city-stratified train/test split (80/20, seed 42, leakage-safe).  
  - `features.py`: feature engineering (log distances, ratios, amenity/luxury flags, buckets) and helper utilities.  
  - `models_regression.py`: regression training/evaluation on `realSum` (log-target training, Euro-scale metrics, baseline, per-city metrics, comparison plots).  
  - `models_classification.py`: high-revenue classification (train-median threshold, city-stratified split, three classifiers, comparison plots).  
  - `models_clustering.py`: KMeans clustering on engineered features (k ∈ {2,3,4}, diagnostics, summaries).  
  - `final_models_switch.py`: orchestrator to run all Step 4 tasks with and without train-only outlier filtering; routes outputs to `plots&models/<With/WithoutOutliers>/<split>/`.

- `plots&models/`  
  All generated artifacts: metrics CSVs, plots, model pickles, terminal logs. Key files include `training_terminal_output.txt`, regression/classification comparison plots, clustering diagnostics, and per-city metrics.

- `info/`  
  - `README.md`: this guide.  
  - `Report.md`: project report (modelling/evaluation narrative).  
  - `requirements.txt`: pinned Python dependencies.

- `src/depricated/`  
  Legacy scripts kept for reference; not used in the current pipeline.
