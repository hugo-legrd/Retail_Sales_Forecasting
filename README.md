# Retail Sales Forecasting Project

A complete data science project that analyzes historical retail sales data and builds predictive models to forecast future sales.

## Table of Contents
- [Project Overview](#project-overview)
- [Data Description](#data-description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [File Structure](#file-structure)
- [Future Improvements](#future-improvements)
- [License](#license)

## Project Overview

This project demonstrates a comprehensive data science workflow for time series forecasting in a retail context. It includes data preparation, exploratory data analysis, feature engineering, model development, evaluation, and deployment as a prediction function.

The goal is to build a model that can accurately predict future retail sales based on historical patterns, holiday effects, promotional periods, and other relevant factors. This type of forecasting is critical for inventory management, staff scheduling, and financial planning in retail businesses.

## Data Description

The project uses synthetic retail sales data that incorporates realistic patterns found in actual retail environments:

- Daily sales data over a 3-year period (2020-2022)
- Upward trending baseline with added yearly and weekly seasonality
- Holiday effects (New Year, Christmas) with pre-holiday build-up
- Promotional periods with sales boosts
- Store closure periods with sales reductions
- Random noise to simulate real-world variability

Key features in the raw data:
- Date
- Daily sales amount
- Day of week indicator
- Month, year, day indicators
- Weekend flag
- Holiday flag
- Promotion flag

## Features

- **Comprehensive data processing pipeline**
- **Extensive feature engineering** for time series data
- **Multiple forecasting models** (Linear Regression, Random Forest)
- **Detailed model evaluation** using MAE, RMSE, and R²
- **Feature importance analysis**
- **Deployment-ready prediction function**
- **Visualizations** of sales patterns and model predictions

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/retail-sales-forecasting.git
cd retail-sales-forecasting

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels

## Usage

### Running the full analysis

```python
python retail_sales_forecasting.py
```

### Using the prediction function

```python
from retail_sales_forecasting import predict_sales

# Example usage
prediction = predict_sales(
    date=pd.Timestamp('2023-01-15'),
    is_weekend=True,
    is_holiday=False,
    is_promotion=True,
    previous_sales={
        'lag_1': 1200,
        'lag_7': 1150,
        'lag_14': 1100,
        'lag_28': 1050,
        'rolling_mean_7': 1175,
        'rolling_mean_14': 1125,
        'rolling_mean_30': 1100,
        'rolling_std_7': 50,
        'rolling_std_14': 75,
        'rolling_std_30': 100
    }
)

print(f"Predicted sales: ${prediction:.2f}")
```

## Methodology

The project follows a structured data science workflow:

1. **Data Collection & Preparation**
   - Generating synthetic data with realistic retail patterns
   - Adding lag features for time series analysis
   - Adding rolling statistics (means, standard deviations)

2. **Exploratory Data Analysis (EDA)**
   - Visualizing sales over time
   - Analyzing day-of-week and monthly patterns
   - Quantifying the impact of holidays and promotions
   - Performing time series decomposition
   - Testing for stationarity

3. **Feature Engineering**
   - Creating cyclical features for month and day of week
   - Adding seasonal indicators (Christmas, summer, etc.)
   - Creating interaction features
   - Generating lag ratio and difference features

4. **Model Development**
   - Training multiple model types
   - Proper time-based train/test splitting
   - Feature scaling
   - Evaluating models on various metrics

5. **Model Evaluation**
   - Visualizing actual vs. predicted values
   - Analyzing feature importances
   - Comparing model performance across metrics

6. **Deployment**
   - Creating a ready-to-use prediction function
   - Saving the best model for production use

## Results

Key findings from the analysis:

- Weekend days show significantly higher sales compared to weekdays
- Promotional periods boost sales by approximately 20-50%
- Holiday periods, especially Christmas, drive substantial sales increases
- Recent sales history (lag features) is the strongest predictor of future sales
- Random Forest outperforms Linear Regression for this dataset
- The model achieves a test R² of approximately 0.90, indicating strong predictive power

## File Structure

```
retail-sales-forecasting/
│
├── retail_sales_forecasting.py   # Main project file
├── models/                      # Saved model files
│   ├── best_sales_model.pkl     # Best performing model
│   └── feature_scaler.pkl       # Feature scaler
│
├── visualizations/              # Generated visualizations
│   ├── sales_over_time.png
│   ├── sales_by_weekday.png
│   ├── sales_by_month.png
│   ├── promotion_effect.png
│   ├── holiday_effect.png
│   ├── weekend_effect.png
│   ├── correlation_matrix.png
│   ├── time_series_decomposition.png
│   ├── feature_importances.png
│   ├── linear_regression_predictions.png
│   └── random_forest_predictions.png
│
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## Future Improvements

Potential enhancements for this project:

- **Advanced Models**: Implement ARIMA, Prophet, or deep learning models
- **Hyperparameter Tuning**: Optimize model parameters
- **External Factors**: Incorporate weather data, economic indicators
- **Anomaly Detection**: Identify and handle outliers in sales data
- **Ensemble Methods**: Combine multiple models for improved accuracy
- **Interactive Dashboard**: Create a web interface for sales forecasting
- **Automated Retraining**: Set up a pipeline for model retraining with new data

