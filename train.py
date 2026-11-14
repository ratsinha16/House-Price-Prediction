# ==========================================
# House Price Prediction
# Complete Training Pipeline
# ==========================================
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)
from sklearn.ensemble import (
    RandomForestRegressor
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import joblib
# ==========================================
# Directories
# ==========================================
os.makedirs("plots", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)
# ==========================================
# Load Dataset
# ==========================================
print("Loading Dataset...")
df = pd.read_csv("train.csv")
print("Dataset Shape:", df.shape)
# ==========================================
# Automated EDA Report
# ==========================================
print("\nGenerating Profiling Report...")
profile = ProfileReport(
    df,
    title="House Price Prediction EDA Report",
    explorative=True
)
profile.to_file(
    "reports/EDA_Report.html"
)
print("EDA Report Saved")
# ==========================================
# Sale Price Distribution
# ==========================================
plt.figure(figsize=(10, 6))
sns.histplot(
    df["SalePrice"],
    kde=True,
    bins=30
)
plt.title("Sale Price Distribution")
plt.savefig(
    "plots/saleprice_distribution.png"
)
plt.close()
# ==========================================
# Missing Values Plot
# ==========================================
missing = (
    df.isnull()
      .sum()
)
missing = missing[
    missing > 0
].sort_values(
    ascending=False
)
plt.figure(figsize=(12, 6))
missing.head(20).plot(
    kind="bar"
)
plt.title(
    "Top Missing Features"
)
plt.tight_layout()
plt.savefig(
    "plots/missing_values.png"
)
plt.close()
# ==========================================
# Feature Selection
# ==========================================
features = [
    "OverallQual",
    "GrLivArea",
    "GarageArea",
    "YearBuilt",
    "FullBath",
    "BedroomAbvGr",
    "TotalBsmtSF",
    "1stFlrSF",
    "TotRmsAbvGrd"
]
target = "SalePrice"
X = df[features]
y = df[target]
# ==========================================
# Heatmap
# ==========================================
heatmap_cols = features + [target]
plt.figure(figsize=(12, 8))
sns.heatmap(
    df[heatmap_cols].corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.savefig(
    "plots/correlation_heatmap.png"
)
plt.close()
# ==========================================
# Missing Value Handling
# ==========================================
imputer = SimpleImputer(
    strategy="median"
)
X = pd.DataFrame(
    imputer.fit_transform(X),
    columns=features
)
# ==========================================
# Train Test Split
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
# ==========================================
# Hyperparameter Tuning
# ==========================================
print("\nStarting GridSearchCV...")
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}
rf = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring="r2",
    cv=5,
    verbose=1,
    n_jobs=-1
)
grid_search.fit(
    X_train,
    y_train
)
model = grid_search.best_estimator_
print("\nBest Parameters:")
print(
    grid_search.best_params_
)
# ==========================================
# Prediction
# ==========================================
predictions = model.predict(
    X_test
)
# ==========================================
# Evaluation
# ==========================================
mae = mean_absolute_error(
    y_test,
    predictions
)
rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)
r2 = r2_score(
    y_test,
    predictions
)
print("\nMODEL PERFORMANCE")
print("-------------------")
print(
    f"MAE  : {mae:.2f}"
)
print(
    f"RMSE : {rmse:.2f}"
)
print(
    f"R²   : {r2:.4f}"
)
# ==========================================
# Actual vs Predicted
# ==========================================
plt.figure(figsize=(8, 6))
plt.scatter(
    y_test,
    predictions,
    alpha=0.6
)
plt.xlabel(
    "Actual Price"
)
plt.ylabel(
    "Predicted Price"
)
plt.title(
    "Actual vs Predicted"
)
plt.savefig(
    "plots/actual_vs_predicted.png"
)
plt.close()
# ==========================================
# Residual Plot
# ==========================================
residuals = (
    y_test - predictions
)
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=predictions,
    y=residuals
)
plt.axhline(
    y=0,
    color="red",
    linestyle="--"
)
plt.title(
    "Residual Analysis"
)
plt.xlabel(
    "Predicted Values"
)
plt.ylabel(
    "Residuals"
)
plt.savefig(
    "plots/residual_analysis.png"
)
plt.close()
# ==========================================
# Feature Importance
# ==========================================
importance_df = pd.DataFrame(
    {
        "Feature": features,
        "Importance":
        model.feature_importances_
    }
)
importance_df = (
    importance_df
    .sort_values(
        by="Importance",
        ascending=False
    )
)
plt.figure(
    figsize=(10, 6)
)
sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature"
)
plt.title(
    "Feature Importance"
)
plt.savefig(
    "plots/feature_importance.png"
)
plt.close()
print("\nFeature Importance")
print(
    importance_df
)
# ==========================================
# Save Model
# ==========================================
joblib.dump(
    model,
    "models/house_price_model.pkl"
)
joblib.dump(
    imputer,
    "models/imputer.pkl"
)
print(
    "\nSaved:"
)
print(
    "models/house_price_model.pkl"
)
print(
    "models/imputer.pkl"
)
# ==========================================
# Sample Prediction
# ==========================================
sample_house = pd.DataFrame([
    {
        "OverallQual": 7,
        "GrLivArea": 1800,
        "GarageArea": 500,
        "YearBuilt": 2010,
        "FullBath": 2,
        "BedroomAbvGr": 3,
        "TotalBsmtSF": 900,
        "1stFlrSF": 1000,
        "TotRmsAbvGrd": 7
    }
])
sample_house = pd.DataFrame(
    imputer.transform(
        sample_house
    ),
    columns=features
)
predicted_price = model.predict(
    sample_house
)
print(
    f"\nPredicted Price: ${predicted_price,.2f}"
)
# ==========================================
# Complete
# ==========================================
print("\nProject Complete!")
print("\nGenerated Files:")
print("""
reports/
└── EDA_Report.html
plots/
├── saleprice_distribution.png
├── missing_values.png
├── correlation_heatmap.png
├── actual_vs_predicted.png
├── residual_analysis.png
└── feature_importance.png
models/
├── house_price_model.pkl
└── imputer.pkl
""")
