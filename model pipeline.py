"""
DSN Bootcamp Qualification Hackathon 2026 - ML Track
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
 
SEED = 42
FOLDS = 5

# 1. LOAD
train = pd.read_csv('/kaggle/input/competitions/dsn-bootcamp-qualification-hackathon-2026-ml-track/train.csv')
test = pd.read_csv('/kaggle/input/competitions/dsn-bootcamp-qualification-hackathon-2026-ml-track/test.csv')
 
train['is_train'] = 1
test['is_train'] = 0
test['total_sales'] = np.nan
df = pd.concat([train, test], ignore_index=True)

# EXPLORATORY DATA ANALYSIS
import matplotlib.pyplot as plt
import seaborn as sns
 
print("train shape:", train.shape)
print("test shape:", test.shape)
print()
 
print("dtypes:")
print(train.dtypes)
print()
 
print("missing values (train):")
print(train.isnull().sum())
print()
print("missing values (test):")
print(test.isnull().sum())
print()
 
# shelf_visibility and store_size stand out as having real gaps, worth digging into
print("missing store_size by store_code - is it random or store-specific?")
print(train[train['store_size'].isnull()]['store_code'].value_counts())
print()
 
print("target distribution:")
print(train['total_sales'].describe())
print()
 
print("fat_content values:")
print(train['fat_content'].value_counts())
print()
 
print("product_category values (note the inconsistent casing):")
print(train['product_category'].value_counts())
print()
 
print("store_format values:")
print(train['store_format'].value_counts())
print()
 
print("store_location_tier values:")
print(train['store_location_tier'].value_counts())
print()
 
# check if shelf_visibility=0 is a real value or a data issue - same product
# should have similar visibility everywhere if 0 is legit
zero_vis_products = train[train['shelf_visibility'] == 0]['product_code'].unique()[:5]
for p in zero_vis_products:
    print(p, "visibility across rows:", train[train['product_code'] == p]['shelf_visibility'].values)
print()
 
# correlation between the numeric features and the target
print("correlation with total_sales:")
print(train[['product_weight_kg', 'shelf_visibility', 'product_price',
             'store_age_years', 'total_sales']].corr()['total_sales'].sort_values(ascending=False))
print()
 
print("average sales by store_format:")
print(train.groupby('store_format')['total_sales'].mean().sort_values())
print()
 
print("average sales by store_location_tier:")
print(train.groupby('store_location_tier')['total_sales'].mean().sort_values())
print()
 
print("average sales by store_size:")
print(train.groupby('store_size', dropna=False)['total_sales'].mean().sort_values())
print()
 
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
 
axes[0, 0].hist(train['total_sales'], bins=50, color='steelblue')
axes[0, 0].set_title('total_sales distribution')
 
sns.boxplot(data=train, x='store_format', y='total_sales', ax=axes[0, 1])
axes[0, 1].set_title('sales by store format')
axes[0, 1].tick_params(axis='x', rotation=30)
 
axes[0, 2].scatter(train['product_price'], train['total_sales'], alpha=0.3, s=10)
axes[0, 2].set_title('price vs sales')
axes[0, 2].set_xlabel('product_price')
axes[0, 2].set_ylabel('total_sales')
 
sns.boxplot(data=train, x='store_location_tier', y='total_sales', ax=axes[1, 0])
axes[1, 0].set_title('sales by location tier')
 
sns.boxplot(data=train, x='store_size', y='total_sales', ax=axes[1, 1])
axes[1, 1].set_title('sales by store size (NaN included)')
 
axes[1, 2].hist(train['shelf_visibility'], bins=50, color='darkorange')
axes[1, 2].set_title('shelf_visibility distribution (note the spike at 0)')
 
plt.tight_layout()
plt.savefig('eda_overview.png', dpi=120)
plt.show()
print("saved eda_overview.png")
 
df['product_category'] = df['product_category'].str.strip().str.title()
 
# shelf_visibility of 0 doesn't make sense for a product actually on a shelf - same
# product shows nonzero visibility elsewhere, so these are basically missing values
df.loc[df['shelf_visibility'] == 0, 'shelf_visibility'] = np.nan
df['shelf_visibility'] = df.groupby('product_category')['shelf_visibility'].transform(lambda x: x.fillna(x.mean()))
 
# weight barely varies for the same product (just measurement noise), so use the
# product's own average weight to fill gaps instead of a generic category average
df['product_weight_kg'] = df.groupby('product_code')['product_weight_kg'].transform(lambda x: x.fillna(x.mean()))
df['product_weight_kg'] = df.groupby('product_category')['product_weight_kg'].transform(lambda x: x.fillna(x.mean()))
 
# store_size is missing for entire stores, not random rows - keep it as its own category
# rather than guessing a value
df['store_size'] = df['store_size'].fillna('Unknown')
 
# a few extra features
df['price_per_kg'] = df['product_price'] / df['product_weight_kg']
 
non_consumables = ['Household', 'Health And Hygiene', 'Others']
drinks = ['Soft Drinks', 'Hard Drinks']
 
def group_category(cat):
    if cat in non_consumables:
        return 'Non-Consumable'
    if cat in drinks:
        return 'Drinks'
    return 'Food'
 
df['broad_category'] = df['product_category'].apply(group_category)
df.loc[df['broad_category'] == 'Non-Consumable', 'fat_content'] = 'Not Applicable'
 
df['store_age_bucket'] = pd.cut(df['store_age_years'], bins=[0, 28, 40, 100], labels=['New', 'Mid', 'Old']).astype(str)
 
df['price_vs_store_avg'] = df['product_price'] / df.groupby('store_code')['product_price'].transform('mean')
df['price_vs_category_avg'] = df['product_price'] / df.groupby('product_category')['product_price'].transform('mean')
 
cat_cols = ['fat_content', 'product_category', 'broad_category', 'store_size',
            'store_location_tier', 'store_format', 'store_age_bucket', 'store_code']
num_cols = ['product_weight_kg', 'shelf_visibility', 'product_price', 'store_age_years',
            'price_per_kg', 'price_vs_store_avg', 'price_vs_category_avg']
 
for c in cat_cols:
    df[c] = df[c].astype(str)
 
train_df = df[df['is_train'] == 1].reset_index(drop=True)
test_df = df[df['is_train'] == 0].reset_index(drop=True)
 
features = cat_cols + num_cols
X = train_df[features]
y = train_df['total_sales']
X_test = test_df[features]
cat_idx = [X.columns.get_loc(c) for c in cat_cols]
 
params = dict(
    iterations=3000,
    learning_rate=0.03,
    depth=4,
    l2_leaf_reg=5,
    random_seed=SEED,
    verbose=False,
)
 
# 5-fold CV to get an honest RMSE estimate and figure out roughly how many
# iterations the model needs before it starts overfitting
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
scores = []
best_iters = []
 
for fold, (tr_idx, val_idx) in enumerate(kf.split(X), start=1):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
 
    model = CatBoostRegressor(**params, cat_features=cat_idx, early_stopping_rounds=100)
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
 
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    scores.append(rmse)
    best_iters.append(model.get_best_iteration())
    print(f"fold {fold}: rmse = {rmse:.2f}, best_iter = {model.get_best_iteration()}")
 
print(f"\naverage CV RMSE: {np.mean(scores):.2f} (+/- {np.std(scores):.2f})")
 
importance = pd.Series(model.get_feature_importance(), index=X.columns).sort_values(ascending=False)
print("\nfeature importance:")
print(importance)
 
# train the final model a few times with different seeds and average the
# predictions - helps smooth out some of the run-to-run variance
n_iters = int(np.mean(best_iters))
seeds = [42, 7, 123, 2024, 99]
test_preds = []
 
for s in seeds:
    m = CatBoostRegressor(**{**params, 'iterations': n_iters, 'random_seed': s}, cat_features=cat_idx)
    m.fit(X, y)
    test_preds.append(m.predict(X_test))
 
final_preds = np.mean(test_preds, axis=0)
 
submission = pd.DataFrame({
    'id': test_df['id'],
    'total_sales': final_preds
})
submission.to_csv('submission.csv', index=False)
 
print("\nsaved submission.csv")
print(submission.head())
print(submission['total_sales'].describe())
 
