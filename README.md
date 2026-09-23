# DSN-Bootcamp-Qualification-Hackathon-2026-ML-Assessment
The DSN Mart Sales Prediction Hackathon is the qualifying competition for the DSN AI Bootcamp. Working with historical retail sales data from DSN Mart to build a machine learning model that predicts the total sales of a product at a specific store. Submitted as part of the qualifying hackathon for the DSN AI Bootcamp.

**Competition:** [DSN Bootcamp Qualification Hackathon 2026 ML Track](https://www.kaggle.com/competitions/dsn-bootcamp-qualification-hackathon-2026-ml-track)
**Task:** Regression, predict `total_sales` for each product-store pair in `test.csv`
**Metric:** RMSE (lower is better)
**Public leaderboard:** 1070.21 RMSE, top 14 of 40

## What's in this repo

- `final_model.py`,the full pipeline: EDA, cleaning, feature engineering, model
  training with cross-validation, and submission generation
- `submission.csv`, the last submitted result predicted
- `test.csv`, Dataset
- `train.csv`, Dataset

## Exploratory data analysis

Before building anything, I looked at:

- **Shape and structure** of train/test, column types, and missing value counts
- **`shelf_visibility = 0`** this turned out to be a data error rather than a real
  value. The same product shows nonzero visibility on other rows, so a value of exactly
  0 doesn't reflect a product that's genuinely invisible on the shelf.
- **`store_size`** is missing for 3 entire stores, not random individual rows, a
  store-level pattern rather than noise, so I kept it as its own `"Unknown"` category
  instead of guessing a value.
- **`product_weight_kg`** varies only ~1.6% for the same product across rows (just
  measurement noise), so missing weights are filled from that product's own average
  rather than a generic category average.
- **`product_category`** had inconsistent casing across rows for the same category,
  normalized before use.
- **Correlations and group averages**: `product_price` is the strongest single numeric
  predictor of sales (0.57 correlation). `store_format` shows a large gap on its own,
  Corner Shops average roughly 6x lower sales than Flagship Hypermarkets. Store location
  tier and size show smaller but visible differences.

Running `final_model.py` prints all of the above and saves `eda_overview.png`, a 6-panel
chart covering the sales distribution, sales by store format/tier/size, price vs. sales,
and the visibility-at-zero issue.

## Cleaning and feature engineering

| Issue | Fix |
|---|---|
| `shelf_visibility == 0` | Treated as missing, imputed with category mean |
| Missing `product_weight_kg` | Imputed from the same product's average weight |
| Missing `store_size` (whole-store pattern) | Kept as explicit `"Unknown"` category |
| Inconsistent `product_category` casing | Normalized to title case |

Added features: `price_per_kg`, a coarser `broad_category` grouping (Food / Drinks /
Non-Consumable), a `store_age_bucket`, and two price-ratio features comparing a
product's price to its store's average and its category's average.

## Modeling

CatBoost was chosen over LightGBM and a linear regression baseline after comparing all
three with 5-fold cross-validation on the same feature set — CatBoost's native handling
of categorical columns outperformed one-hot encoding and edged out LightGBM. A shallow
tree (`depth=4`) generalized better than deeper trees, which fits given the training set
is relatively small (~6,800 rows).

| Model | CV RMSE |
|---|---|
| Linear Regression (baseline) | 1130.12 |
| LightGBM | 1082.77 |
| **CatBoost, tuned (depth=4, lr=0.03, l2=5)** | **1074.63** |

The final model trains 5 CatBoost models with different random seeds and averages their
predictions, which reduces prediction variance without adding any new features.

**Top predictive features:** `store_format`, `price_vs_store_avg`, `product_price`,
`price_vs_category_avg`.

## Results

| Metric | Score |
|---|---|
| Cross-validation RMSE | 1074.63 |
| Public leaderboard RMSE | **1070.21** |
| Public leaderboard rank | Top 14 of 40 |

## What I tried that didn't help

An earlier version added more engineered features (product/store count features,
target-encoded categoricals) and combined three models into an ensemble. It scored
better in cross-validation (1077.20) but worse on the actual leaderboard (1079.94).
Digging into why: one of the added features, a count of products per store, turned out
to be nearly constant across stores, so it added noise rather than real signal. That's
the reason the final version here sticks to a smaller, individually-validated set of
features rather than everything I tried along the way, on a dataset this size, added
complexity needs to earn its place feature by feature, not all at once.

## How to run

```bash
pip install -r requirements.txt
python final_model.py
```
