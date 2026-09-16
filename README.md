# DSN-Bootcamp-Qualification-Hackathon-2026-ML-Assessment
The DSN Mart Sales Prediction Hackathon is the qualifying competition for the DSN AI Bootcamp. Working with historical retail sales data from DSN Mart to build a machine learning model that predicts the total sales of a product at a specific store. Submitted as part of the qualifying hackathon for the DSN AI Bootcamp.

**Competition:** [DSN Bootcamp Qualification Hackathon 2026 ML Track](https://www.kaggle.com/competitions/dsn-bootcamp-qualification-hackathon-2026-ml-track)
**Task:** Regression predict `total_sales` for each product-store pair in `test.csv`
**Metric:** RMSE (lower is better)
**Final leaderboard score:** 1070.77 RMSE

## Approach summary

The pipeline follows five stages: clean -> engineer features -> validate with cross-validation
-> compare models -> generate predictions. Every design decision below was checked against
5-fold cross-validation before being kept, and several were later checked against real
Kaggle leaderboard feedback.

## 1. Exploratory data analysis: key findings

- **`shelf_visibility = 0` is a data error, not a real value.** The same product shows
  nonzero visibility on other rows, confirming 0 means missing rather than "invisible on
  the shelf." Treated as missing and imputed.
- **`store_size` is missing for 3 entire stores**, not randomly across rows. This is
  informative (those stores simply never report a size), so it's kept as its own
  `"Unknown"` category rather than imputed with a guessed value.
- **`product_weight_kg` is near-constant per product** (varies ~1.6% due to measurement
  noise), so missing weights are recovered from that product's own average weight across
  train + test, rather than a generic category average.
- **`product_category` had inconsistent casing** across rows (e.g. multiple casing
  variants of the same category name) normalized before use.
- **`product_price` is by far the strongest single predictor** (0.57 correlation with
  sales). `store_format` also shows a large gap on its own, Corner Shops average roughly
  6x lower sales than Flagship Hypermarkets.
- `product_weight_kg` and `store_age_years` show almost no *linear* correlation with
  sales on their own, but still contribute through tree-based interactions.

## 2. Data cleaning

| Issue | Fix |
|---|---|
| `shelf_visibility == 0` | Treated as missing, imputed with category mean |
| Missing `product_weight_kg` | Imputed from the same product's average weight |
| Missing `store_size` (whole-store pattern) | Kept as explicit `"Unknown"` category |
| Inconsistent `product_category` casing | Normalized to title case |

## 3. Feature engineering

- `price_per_kg` - price relative to weight
- `broad_category` - Food / Drinks / Non-Consumable grouping
- `store_age_bucket` - New / Mid / Old store age buckets
- `fat_content` normalized to `"Not Applicable"` for non-consumables (household goods
  don't meaningfully have a "low fat" attribute)

## 4. Modeling

LightGBM was compared against a linear regression baseline via 5-fold cross-validation,
using native categorical handling (pandas `category` dtype) rather than one-hot encoding,
and predicting the raw sales value directly rather than a log-transformed target — both
choices were tested and confirmed better empirically.

| Model | CV RMSE |
|---|---|
| Linear Regression (baseline) | 1130.12 |
| **LightGBM (final model)** | **1082.77** |

**Top predictive features:** `product_price`, `product_weight_kg`, `price_per_kg`,
`store_format`, `store_code`.

## 5. Results

| Metric | Score |
|---|---|
| Cross-validation RMSE | 1082.77 |
| Public leaderboard RMSE | **1070.77** |
| Public leaderboard rank | Top 8 of 40 |

The leaderboard score landing slightly *better* than the CV estimate is a good sign —
it means the model generalizes well and isn't overfitting to the training data.

## A note on further experiments

A follow-up attempt combined extra engineered features (product/store count features,
target-encoded categoricals) with an ensemble of LightGBM, XGBoost, and CatBoost. This
scored *better* in cross-validation (1077.20) but *worse* on the real leaderboard
(1079.94) than the simpler LightGBM model above. Investigating why: one of the new
features (`store_product_count`) turned out to be almost constant across stores — every
store carries roughly the same number of products, so it added noise rather than signal.
A separate, more careful pass using CatBoost alone with the *original* clean feature set
(no noisy extras) reached a CV RMSE of 1074.63 after tuning, but wasn't leaderboard-tested
before this LightGBM submission was locked in as final. This is a useful reminder that
on a small dataset (~6,800 training rows), added complexity should be validated
individually rather than combined all at once, and that cross-validation gains don't
always translate directly to leaderboard gains.

## How to run

```bash
pip install -r requirements.txt
python model_pipeline.py
```

Expects `train.csv` and `test.csv` from the competition's Kaggle data page. Update the
file paths at the top of the script to match your environment (Kaggle notebook input
path, or a local path if running elsewhere). Outputs `submission.csv` in the exact format
required by the competition.

## Key takeaways

- Cross validation is the right compass, but a large public-vs-real-leaderboard gap is
  a signal worth investigating rather than ignoring.
- On a small dataset, simpler and better-validated beats more complex and untested.
- Store format and product price dominate what drives sales here, a useful business
  insight on top of the leaderboard score itself.
