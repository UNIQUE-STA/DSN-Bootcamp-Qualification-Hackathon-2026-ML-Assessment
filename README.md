# DSN-Bootcamp-Qualification-Hackathon-2026-ML-Assessment
The DSN Mart Sales Prediction Hackathon is the qualifying competition for the DSN AI Bootcamp. Working with historical retail sales data from DSN Mart to build a machine learning model that predicts the total sales of a product at a specific store. 

# Key
Predicting total product sales for DSN Mart, a retail chain across Nigeria, based on
product and store attributes. Submitted as part of the qualifying hackathon for the
DSN AI Bootcamp.

**Competition:** [DSN Bootcamp Qualification Hackathon 2026 ML Track](https://www.kaggle.com/competitions/dsn-bootcamp-qualification-hackathon-2026-ml-track)
**Task:** Regression — predict `total_sales` for each product-store pair in `test.csv`
**Metric:** RMSE (lower is better)

## Approach summary

The pipeline follows five stages: clean → engineer features → validate with cross-validation
→ compare models → generate predictions. Every design decision below was checked against
5-fold cross-validation before being kept, and several were later checked against real
Kaggle leaderboard feedback.

## 1. Exploratory data analysis — key findings

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
  variants of the same category name) — normalized before use.
- **`product_price` is by far the strongest single predictor** (0.57 correlation with
  sales). `store_format` also shows a large gap on its own — Corner Shops average roughly
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

- `price_per_kg` — price relative to weight
- `broad_category` — Food / Drinks / Non-Consumable grouping
- `store_age_bucket` — New / Mid / Old store age buckets
- `price_vs_store_avg` — how a product's price compares to the average at its store
- `price_vs_category_avg` — how a product's price compares to others in its category
- `fat_content` normalized to `"Not Applicable"` for non-consumables (household goods
  don't meaningfully have a "low fat" attribute)

**A note on what didn't work:** an earlier version added extra features
(`product_store_count`, `store_product_count`, target-encoded `store_code` /
`product_category` / a format-tier interaction) and combined three models (LightGBM,
XGBoost, CatBoost) into an averaged ensemble. This scored *better* in cross-validation
(1077.20 vs. 1082.77) but *worse* on the actual leaderboard (1079.94 vs. 1070.77).
Investigating why: `store_product_count` turned out to be almost constant across stores
(every store carries roughly the same number of products), so it was adding noise rather
than signal — a good reminder that CV improvements need to be interpreted cautiously on
small datasets (~6,800 training rows), and that every new feature should be tested
individually rather than combined all at once.

## 4. Modeling

Three gradient boosting models were compared via 5-fold cross-validation on the same
clean feature set:

| Model | CV RMSE |
|---|---|
| Linear Regression (baseline) | 1130.12 |
| LightGBM | 1082.77 |
| CatBoost | 1076.06 |
| CatBoost (tuned: depth=4, lr=0.03, l2=5) | **1074.63** |

CatBoost's native categorical handling outperformed one-hot encoding and outperformed
LightGBM on this dataset. A shallower tree (`depth=4`) generalized better than deeper
trees, consistent with the small training set size.

**Final model:** CatBoost with tuned hyperparameters, seed-bagged across 5 random seeds
(the final predictions are the average of 5 models trained with different seeds) to
reduce prediction variance without adding any new features or overfitting risk.

**Top predictive features:** `product_price`, `store_format`, `store_age_years`,
`price_per_kg`, `store_code`.

## 5. Results

| Submission | CV RMSE | Public Leaderboard RMSE |
|---|---|---|
| LightGBM, clean features | 1082.77 | 1070.77 |
| Ensemble + extra features (regressed) | 1077.20 | 1079.94 |
| CatBoost, clean features | 1076.06 | Improved rank into top 8 of 40 |
| CatBoost, tuned + seed-bagged | 1074.63 | *pending* |

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

- Cross-validation is the right compass, but a large public-vs-real-leaderboard gap is
  a signal worth investigating rather than ignoring — in this case it revealed a genuinely
  noisy feature.
- On a small dataset, simpler and better-validated beats more complex and untested.
- Store format and product price dominate what drives sales here — a useful business
  insight on top of the leaderboard score itself.
