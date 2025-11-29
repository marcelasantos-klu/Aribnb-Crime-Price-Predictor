
## Objective 
Overview the Airbnb bookings and crime and safety indices in those cities. We found both datasets in Kaggle: one for Airbnb bookings (sales, room type, city) and one for crime and safety indices (city and indices), therefore, the city feature will be a key that links the two datasets. 

The model is made predict the revenue of a booking based on the specific type of accommodation, city, and crime/safety conditions. 


Project tasks (aligned with the assignment guide):
1) Business and data understanding
- Define the core business question (e.g., demand forecast, price elasticity, margin optimization) and target variable(s).
- Summarize the dataset: entities, granularity, time span, and each column’s role; note any external sources you plan to add (e.g., holidays, weather, promotions).
- Run EDA to see distributions, correlations, seasonality, and how features relate to the target; identify potential segments (e.g., category, brand, collection).

2) Data preparation
- Clean data: handle missing values, duplicates, outliers, data types, and currency/units; document any rows removed.
- Encode/scale/normalize as needed; create train/validation/test splits that respect time (if applicable) and balance classes if classification.
- Engineer features relevant to pricing and demand (e.g., discount depth, time since launch, season, stock position, competitor price proxies, lag/rolling stats).

3) Modeling
- Establish a simple baseline (e.g., mean/median, naive seasonal forecast, logistic baseline) for comparison.
- Train multiple model families with parameter variation (e.g., linear/elastic net, tree-based models such as RF/GBM/XGBoost, regularized GLMs, and, if useful, a time-series or panel model). Consider an unsupervised step (clustering segments) to inform features or pricing groups.
- Track experiments and hyperparameters; compare how feature choices and model classes affect performance.

4) Evaluation
- Use suitable metrics for the task (e.g., MAE/RMSE/SMAPE for regression; accuracy/F1/ROC-AUC/PR-AUC for classification; business KPIs such as expected margin or stock-out risk).
- Apply cross-validation or time-based splits; highlight how evaluation method changes results. Include calibration checks and error analysis (by category/season/price band).
- Present performance against the baseline and between models; explain trade-offs and when each model is preferable.

5) Documentation and reflection
- Keep code commented and reproducible (scripts or notebook) with clear step ordering.
- Write up decisions, assumptions, and limitations; note issues like data leakage risks, feature availability at prediction time, and fairness/segment impacts.
- Reflect on how good the final model is, what else could be done next, and lessons learned.

Deliverables (per guide)
- Presentation (~5 minutes): problem, dataset, approach, key results, and takeaways.
- Python implementation: executable code plus required data/notebook; comments explaining each step.
- Report (about 3–4 pages per team member): cover all tasks above with plots/graphs for data, analyses, and results; include reflection on performance and next steps.
- Oral defense readiness: every team member can explain and justify choices across data prep, modeling, and evaluation.

Checklist to satisfy assessment criteria
- Problem is well framed in both domain and technical terms; dataset complexity is appropriate.
- Data is properly prepared (encodings, balancing, handling NaNs/outliers) and documented.
- Techniques and evaluation metrics fit the problem; multiple models and parameter variations tested.
- Choices are explained in the report; structure and flow are clear.
- Team can justify design decisions and discuss limitations in an oral defense.


