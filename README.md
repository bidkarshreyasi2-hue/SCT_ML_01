# SCT_ML_01
# House Price Prediction 🏠

A machine learning project that predicts residential house sale prices
using **Linear Regression**, based on total living area, number of
bedrooms, and number of bathrooms.

## Dataset

[House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) (Kaggle)

The dataset contains 79 features describing residential homes in Ames,
Iowa. This project focuses on three engineered features:

TotalSF – total square footage (basement + 1st floor + 2nd floor)
TotalBathrooms – full baths + 0.5 × half baths (basement included)
Bedrooms – number of bedrooms above grade
OverallQual – overall material and finish quality (1–10)
OverallCond – overall condition rating (1–10)
HouseAge – years between construction and sale (YrSold - YearBuilt)
YearsSinceRemodel – years since last remodel at time of sale
GarageCars – size of garage in car capacity
LotArea – lot size in square feet
Fireplaces – number of fireplaces
Neighborhood – categorical, one-hot encoded

## Approach

1.Load train.csv/test.csv
2.EDA: target distribution, missing values, correlations
3.Engineer features (handle missing values)
4.Build preprocessing pipeline (scale numeric, one-hot encode Neighborhood)
5.Log-transform target, fit models
6.Compare Linear Regression, Ridge, Random Forest via 5-fold CV
7.Evaluate chosen model on 80/20 validation split (RMSE, MAE, R²)
8.Inspect coefficients and residuals
9.Plot actual vs. predicted
10.Refit on full data, generate submission.csv

## Results

validation split:

| Metric | Value |
|--------|-------|
| RMSE   | ~45,800 |
| MAE    | ~29,400 |
| R²     | ~0.73 |

## Project structure

```
house-price-prediction/
├── data/
│   ├── train.csv
│   └── test.csv
├── house_price_prediction.py
├── requirements.txt
└── README.md
```



## Tech stack

- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
