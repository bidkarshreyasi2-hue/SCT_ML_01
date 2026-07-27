# SCT_ML_01
# House Price Prediction 🏠

A machine learning project that predicts residential house sale prices
using **Linear Regression**, based on total living area, number of
bedrooms, and number of bathrooms.

## Dataset

[House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) (Kaggle)

The dataset contains 79 features describing residential homes in Ames,
Iowa. This project focuses on three engineered features:

- **TotalSF** – total square footage (basement + 1st floor + 2nd floor)
- **TotalBathrooms** – full baths + 0.5 × half baths (basement included)
- **Bedrooms** – number of bedrooms above grade

## Approach

1. Load `train.csv` and `test.csv`.
2. Engineer the three features above (handling missing values).
3. Split the training data 80/20 for training and validation.
4. Fit a `LinearRegression` model from scikit-learn.
5. Evaluate on the validation set (RMSE, MAE, R²).
6. Visualize actual vs. predicted prices.
7. Generate predictions for the test set (`submission.csv`).

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

## How to run

```bash
pip install -r requirements.txt
python house_price_prediction.py
```

This will print validation metrics, save `actual_vs_predicted.png`,
and write predictions to `submission.csv`.

## Tech stack

- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
