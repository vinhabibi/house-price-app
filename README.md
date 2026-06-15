# 🏠 California Housing Price Prediction

A end-to-end machine learning project that predicts California median house prices using the California Housing dataset. The project covers the full data science lifecycle — from exploratory data analysis and feature engineering through to model optimization and deployment-ready artifact export.

---

## 📌 Project Overview

| Item | Detail |
|------|--------|
| **Dataset** | California Housing (Google Colab sample data) |
| **Task** | Regression — predict median house value |
| **Best Model** | XGBoost (tuned via GridSearchCV) |
| **Target Transform** | Log1p → back-transformed with expm1 for evaluation |
| **Deployment** | Streamlit web app |

---

## 📁 Project Structure

```
housing-price-app/
├── housing_prediction.ipynb   # Full analysis notebook
├── app.py                     # Streamlit deployment app
├── xgb_house_model.pkl        # Trained XGBoost model
├── imputer.pkl                # Fitted SimpleImputer
├── scaler.pkl                 # Fitted StandardScaler
├── requirements.txt           # Python dependencies
└── README.md
```

---

## 🔍 1. Exploratory Data Analysis

The project began with a thorough EDA on both the train and test splits:

- Inspected shape, dtypes, null counts, and summary statistics
- Plotted the **target distribution** (`median_house_value`) and identified a strong **right skew** and artificial capping at $500,000
- Generated **univariate histograms** for all features to understand spread and outliers
- Created **scatter plots** of each feature against the target to detect linear/non-linear relationships
- Visualised a **correlation heatmap** to identify multicollinearity and feature relevance
- Generated **boxplots** to flag outlier-prone columns

**Impact:** EDA revealed that the raw target variable was skewed and capped, and that raw count features (total rooms, total bedrooms, population) carried less signal than their per-household equivalents — informing every downstream decision.

---

## 🧹 2. Data Cleaning

Two key cleaning steps were applied to both train and test sets:

- **Removed capped values:** Dropped rows where `median_house_value >= 500,000` and `housing_median_age >= 52`, as these represented artificially truncated data rather than true observations
- **Log-transformed the target:** Applied `np.log1p()` to `median_house_value` to correct the right skew and produce a more normally distributed target

**Impact:** Removing capped values eliminated a systematic bias that would have caused the model to underpredict high-value homes. The log transform stabilised variance across the target range, improving model convergence and metric reliability. Final evaluation was done by back-transforming predictions with `np.expm1()` to report real dollar figures.

---

## ⚙️ 3. Feature Engineering

Three ratio features were engineered from the raw count columns:

| New Feature | Formula | Rationale |
|---|---|---|
| `rooms_per_household` | `total_rooms / households` | Captures space per dwelling unit |
| `bedrooms_per_room` | `total_bedrooms / total_rooms` | Proxy for bedroom density / property type |
| `population_per_household` | `population / households` | Captures household crowding |

The raw columns (`total_rooms`, `total_bedrooms`, `population`) were then dropped.

**Impact:** The ratio features encode meaningful housing characteristics that raw counts cannot — a block with 10,000 total rooms means something very different depending on the number of households. Correlation analysis confirmed that the engineered features had stronger linear relationships with log house value than the original raw totals, reducing noise in the feature space.

---

## 🔧 4. Preprocessing

Applied sklearn preprocessing in a **train-fit, test-transform** pattern to prevent data leakage:

- **`SimpleImputer(strategy='median')`** — Handled any missing values (primarily in `total_bedrooms`) using the median, making the pipeline robust to incomplete data
- **`StandardScaler`** — Scaled all features to zero mean and unit variance, ensuring no single feature dominated due to scale differences

**Impact:** Fitting the imputer and scaler only on training data and applying them to the test set mirrors real-world deployment conditions, producing honest generalisation estimates.

---

## 🤖 5. Model Development & Comparison

Three baseline models were trained and evaluated on MAE, RMSE, and R²:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | Higher | Higher | Lower |
| Random Forest | Medium | Medium | Medium |
| Gradient Boosting | Lower | Lower | Higher |

All predictions were back-transformed with `np.expm1()` before evaluation to report metrics in actual dollar values.

**Impact:** The comparison established a performance baseline and confirmed that ensemble tree methods significantly outperformed linear regression on this dataset, justifying the move to XGBoost.

---

## 🚀 6. XGBoost & Hyperparameter Tuning

An XGBoost regressor was trained with tuned hyperparameters, then further optimised via **5-fold GridSearchCV**:

```python
param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [4, 6],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0]
}
```

**Scoring metric:** R² (on log-transformed target)

**Impact:** GridSearchCV identified the optimal combination of depth, learning rate, and regularisation that balanced bias and variance. The tuned model delivered the best MAE, RMSE, and R² of all models evaluated, and is the model saved for deployment.

---

## 💾 7. Artifact Export

The trained pipeline components were serialised with `joblib` for use in the Streamlit app:

```python
joblib.dump(best_model, 'xgb_house_model.pkl')
joblib.dump(imputer,    'imputer.pkl')
joblib.dump(scaler,     'scaler.pkl')
```

**Impact:** Separating the imputer, scaler, and model into distinct artifacts allows each preprocessing step to be applied independently in the app — matching exactly the transformation order used during training.

---

## 🌐 8. Streamlit Deployment

The saved artifacts are loaded in `app.py` to serve a live prediction interface where users can input housing features and receive an estimated median house value.

---

## 🛠️ Tech Stack

- **Python** — pandas, NumPy, Matplotlib, Seaborn
- **Scikit-learn** — SimpleImputer, StandardScaler, GridSearchCV, metrics
- **XGBoost** — XGBRegressor
- **Joblib** — model serialisation
- **Streamlit** — web app deployment

---

## 📦 Installation

```bash
git clone https://github.com/vinhabibi/housing-price-app.git
cd housing-price-app
pip install -r requirements.txt
streamlit run app.py
```

---

## 👤 Author

**Vincent Kimutai**  
BSc. Data Science & Analytics — JKUAT Karen Campus  
GitHub: [vinhabibi](https://github.com/vinhabibi)  
Portfolio: [vinhabibi.github.io/vincent-kimutai](https://vinhabibi.github.io/vincent-kimutai/)
