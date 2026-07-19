# Customer Churn Prediction

Predicting which telecom customers are likely to churn (cancel their subscription), so the business can intervene before they leave.

## Problem Statement

Customer churn is expensive — acquiring a new customer typically costs far more than retaining an existing one. This project builds a machine learning pipeline that flags at-risk customers using their account, billing, and service usage data, so retention teams can act proactively instead of reactively.

## Dataset

- **Source:** [IBM Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn) (public sample dataset)
- **Size:** 7,043 customers, 21 features
- **Target:** `Churn` (Yes/No)
- **Features:** demographics (gender, senior citizen, partner, dependents), account info (tenure, contract type, payment method), services subscribed (internet, phone, streaming, security, backup), and billing (monthly/total charges)

## Approach

1. **EDA** — explored churn distribution and how tenure, charges, contract type, and internet service relate to churn
2. **Preprocessing** — cleaned `TotalCharges`, label-encoded categorical features, scaled numeric features
3. **Class imbalance** — the dataset is ~73/27 skewed toward "No Churn"; addressed with **SMOTE** on the training split only
4. **Modeling** — trained and compared three classifiers:
   - Logistic Regression (interpretable baseline)
   - Decision Tree
   - Random Forest
5. **Evaluation** — accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, and ROC curves for all three models
6. **Feature importance** — identified the strongest churn drivers using the Random Forest model

## Results

| Model               | Accuracy | Precision | Recall | F1   | ROC-AUC |
|---------------------|:--------:|:---------:|:------:|:----:|:-------:|
| Logistic Regression | 0.73     | 0.50      | 0.76   | 0.60 | 0.83    |
| Decision Tree        | 0.75     | 0.51      | 0.77   | 0.62 | 0.82    |
| Random Forest        | 0.77     | 0.55      | 0.65   | 0.59 | 0.82    |

**F1-score** was prioritized over raw accuracy, since missing an actual churner (false negative) is costlier to the business than flagging a loyal customer by mistake.

### Key churn drivers
- Low **tenure** (newer customers churn more)
- **Month-to-month contracts** (vs. 1-2 year contracts)
- Higher **monthly charges**
- **Fiber optic** internet service

## Business Takeaways

- Offering incentives to move month-to-month customers onto longer contracts could meaningfully reduce churn.
- New customers (low tenure) are the highest-risk segment — a stronger onboarding or early-loyalty program is worth testing.
- Investigate why fiber optic customers churn more — likely pricing or service quality issues.

## Tech Stack

- Python, Pandas, NumPy
- scikit-learn (Logistic Regression, Decision Tree, Random Forest)
- imbalanced-learn (SMOTE)
- Matplotlib, Seaborn

## How to Run

1. Clone this repo
2. Open `Customer_Churn_Prediction.ipynb` in Jupyter or Google Colab
3. Run all cells top to bottom (dataset is loaded directly from a public URL — no manual download needed)

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```

## Possible Extensions

- Try XGBoost / LightGBM for further performance gains
- Hyperparameter tuning with GridSearchCV
- Deploy the trained model as a simple Streamlit app for interactive scoring

## Author

Nirmal — B.Tech CSE (Data Science), D Y Patil International University
