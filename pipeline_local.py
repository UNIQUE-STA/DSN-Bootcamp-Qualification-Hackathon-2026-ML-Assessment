"""
DSN Bootcamp Qualification Hackathon 2026 - ML Track
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

RANDOM_STATE = 42
N_FOLDS = 5

# ---------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------
train = pd.read_csv('C:/Users/LENOVO L440/Downloads/train.csv')
test = pd.read_csv('C:/Users/LENOVO L440/Downloads/test.csv')

train['is_train'] = 1
test['is_train'] = 0
test['total_sales'] = np.nan
full = pd.concat([train, test], ignore_index=True)

# ---------------------------------------------------------
# 2. CLEAN
# ---------------------------------------------------------
# Fix inconsistent casing in product_category (e.g. "Snack Foods" vs "SNACK FOODS" vs "snack foods")
full['product_category'] = full['product_category'].str.strip().str.title()

# shelf_visibility == 0 is a data error (same product has nonzero visibility elsewhere) -> treat as missing
full.loc[full['shelf_visibility'] == 0, 'shelf_visibility'] = np.nan

# Impute shelf_visibility using the mean visibility for that product_category
full['shelf_visibility'] = full.groupby('product_category')['shelf_visibility'] \
    .transform(lambda x: x.fillna(x.mean()))

# Impute product_weight_kg using the mean weight of that exact product_code (weight is ~constant per product,
# varies <2% due to measurement noise). Fall back to category mean for the few products with no weight data at all.
full['product_weight_kg'] = full.groupby('product_code')['product_weight_kg'] \
    .transform(lambda x: x.fillna(x.mean()))
full['product_weight_kg'] = full.groupby('product_category')['product_weight_kg'] \
    .transform(lambda x: x.fillna(x.mean()))

# store_size is missing entirely for 3 specific stores (not random) -> keep as its own explicit category
full['store_size'] = full['store_size'].fillna('Unknown')

# ---------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------
# Price per kg - how expensive the product is relative to its weight
full['price_per_kg'] = full['product_price'] / full['product_weight_kg']

# Broad category grouping (Food vs Drinks vs Non-Consumable) - captures a coarser pattern than 16 categories
non_consumable = ['Household', 'Health And Hygiene', 'Others']
drinks = ['Soft Drinks', 'Hard Drinks']
def broad_category(cat):
    if cat in non_consumable:
        return 'Non-Consumable'
    elif cat in drinks:
        return 'Drinks'
    else:
        return 'Food'
full['broad_category'] = full['product_category'].apply(broad_category)

# Non-consumables logically shouldn't have a meaningful fat_content -> normalize it
full.loc[full['broad_category'] == 'Non-Consumable', 'fat_content'] = 'Not Applicable'

# Store age bucket (young / mid / old) - captures nonlinear effects simply
full['store_age_bucket'] = pd.cut(full['store_age_years'], bins=[0, 28, 40, 100],
                                    labels=['New', 'Mid', 'Old'])

categorical_cols = ['fat_content', 'product_category', 'broad_category', 'store_size',
                    'store_location_tier', 'store_format', 'store_age_bucket', 'store_code']
numeric_cols = ['product_weight_kg', 'shelf_visibility', 'product_price',
                'store_age_years', 'price_per_kg']

# LightGBM handles categoricals natively (and better than one-hot encoding, tested empirically) -
# just needs pandas 'category' dtype rather than dummy columns.
for c in categorical_cols:
    full[c] = full[c].astype('category')

feature_cols = categorical_cols + numeric_cols
train_enc = full[full['is_train'] == 1].copy()
test_enc = full[full['is_train'] == 0].copy()

X = train_enc[feature_cols]
y = train_enc['total_sales']
X_test = test_enc[feature_cols]

# ---------------------------------------------------------
# 5. CROSS-VALIDATED BASELINE (Linear Regression, one-hot encoded)
# ---------------------------------------------------------
X_lr = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
lr_rmses = []
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_lr)):
    X_tr, X_val = X_lr.iloc[tr_idx], X_lr.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    model = LinearRegression()
    model.fit(X_tr, y_tr)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    lr_rmses.append(rmse)
print(f"[Linear Regression] CV RMSE: {np.mean(lr_rmses):.2f} (+/- {np.std(lr_rmses):.2f})")

# ---------------------------------------------------------
# 6. CROSS-VALIDATED LightGBM (native categorical handling, raw target)
# Tested: raw target beat log1p target here; native categorical beat one-hot for LightGBM.
# ---------------------------------------------------------
lgb_params = dict(
    n_estimators=2000,
    learning_rate=0.02,
    num_leaves=7,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.7,
    random_state=RANDOM_STATE,
    verbosity=-1,
)

lgb_rmses = []
best_iters = []
for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        categorical_feature=categorical_cols,
        callbacks=[lgb.early_stopping(80, verbose=False)],
    )
    preds = model.predict(X_val, num_iteration=model.best_iteration_)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    lgb_rmses.append(rmse)
    best_iters.append(model.best_iteration_)
    print(f"  Fold {fold+1}: RMSE = {rmse:.2f}  (best_iter={model.best_iteration_})")

print(f"[LightGBM] CV RMSE: {np.mean(lgb_rmses):.2f} (+/- {np.std(lgb_rmses):.2f})")

# ---------------------------------------------------------
# 7. FEATURE IMPORTANCE (from last fold model, for insight)
# ---------------------------------------------------------
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 most important features (LightGBM):")
print(importances.head(10))

# ---------------------------------------------------------
# 8. TRAIN FINAL MODEL ON ALL TRAINING DATA
# Use the average best_iteration across CV folds as a fixed tree count (no validation set left to early-stop on).
# ---------------------------------------------------------
final_n_estimators = int(np.mean(best_iters))
final_params = {**lgb_params, 'n_estimators': final_n_estimators}
final_model = lgb.LGBMRegressor(**final_params)
final_model.fit(X, y, categorical_feature=categorical_cols)

test_preds = final_model.predict(X_test)

# ---------------------------------------------------------
# 9. BUILD SUBMISSION
# ---------------------------------------------------------
submission = pd.DataFrame({
    'id': test_enc['id'].values,
    'total_sales': test_preds
})
submission.to_csv('C:/Users/LENOVO L440/Downloads/submission.csv', index=False)
print("\nSubmission saved. Preview:")
print(submission.head())
print("\nPrediction stats:")
print(submission['total_sales'].describe())
