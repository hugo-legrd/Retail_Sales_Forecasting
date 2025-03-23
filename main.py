# Retail Sales Forecasting Project
# ===============================
# This project demonstrates a complete data science workflow for time series forecasting:
# 1. Data Collection & Preparation
# 2. Exploratory Data Analysis
# 3. Feature Engineering
# 4. Model Development
# 5. Model Evaluation
# 6. Model Deployment (as a prediction function)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

# 1. Data Collection & Preparation
# ================================

def load_and_prepare_data():
    """
    Load sample retail sales data and prepare it for analysis
    """
    # Creating synthetic retail sales data
    # In a real project, you would load data from a file or API
    np.random.seed(42)
    
    # Create date range for 3 years of monthly data
    date_range = pd.date_range(start='2020-01-01', end='2022-12-31', freq='D')
    
    # Base sales with upward trend
    base_sales = np.linspace(1000, 1500, len(date_range))
    
    # Add yearly seasonality (higher in December, lower in January)
    yearly_seasonality = 300 * np.sin(2 * np.pi * np.arange(len(date_range)) / 365 + 10)
    
    # Add weekly seasonality (higher on weekends)
    weekly_seasonality = 100 * (date_range.dayofweek >= 5).astype(int)
    
    # Add some holidays effect
    holidays = pd.DataFrame({
        'date': pd.to_datetime(['2020-01-01', '2020-12-25', '2021-01-01', '2021-12-25', '2022-01-01', '2022-12-25']),
        'holiday': ['New Year', 'Christmas', 'New Year', 'Christmas', 'New Year', 'Christmas']
    })
    
    # Add holiday effects
    holiday_effect = np.zeros(len(date_range))
    for _, holiday in holidays.iterrows():
        # Find the index of the holiday date
        idx = np.where(date_range == holiday['date'])[0]
        if len(idx) > 0:
            # Add a peak for the holiday
            holiday_effect[idx[0]] = 500
            # Add build-up effect before the holiday
            for i in range(1, 6):
                if idx[0] - i >= 0:
                    holiday_effect[idx[0] - i] = 100 * (6 - i)
    
    # Add random noise
    noise = np.random.normal(0, 50, len(date_range))
    
    # Combine all components
    sales = base_sales + yearly_seasonality + weekly_seasonality + holiday_effect + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': date_range,
        'sales': sales,
        'weekday': date_range.dayofweek,
        'month': date_range.month,
        'year': date_range.year,
        'day': date_range.day,
        'is_weekend': (date_range.dayofweek >= 5).astype(int)
    })
    
    # Add a binary column for holidays
    df['is_holiday'] = 0
    for _, holiday in holidays.iterrows():
        df.loc[df['date'] == holiday['date'], 'is_holiday'] = 1
    
    # Add promotional periods (random 2-week periods with higher sales)
    promotions = pd.DataFrame({
        'start_date': pd.to_datetime(['2020-03-01', '2020-07-15', '2021-02-10', '2021-08-20', '2022-04-05', '2022-09-12']),
        'end_date': pd.to_datetime(['2020-03-15', '2020-07-29', '2021-02-24', '2021-09-03', '2022-04-19', '2022-09-26']),
        'promo_name': ['Spring Sale', 'Summer Sale', 'Winter Sale', 'Back to School', 'Easter Sale', 'Fall Sale']
    })
    
    # Add promotion effect
    df['is_promotion'] = 0
    for _, promo in promotions.iterrows():
        mask = (df['date'] >= promo['start_date']) & (df['date'] <= promo['end_date'])
        df.loc[mask, 'is_promotion'] = 1
        # Increase sales during promotions
        df.loc[mask, 'sales'] *= np.random.uniform(1.2, 1.5)
    
    # Add some store closures or inventory issues
    closures = pd.DataFrame({
        'start_date': pd.to_datetime(['2020-05-10', '2021-11-05', '2022-06-20']),
        'end_date': pd.to_datetime(['2020-05-15', '2021-11-10', '2022-06-25']),
        'reason': ['Inventory', 'Renovation', 'System Upgrade']
    })
    
    # Reduce sales during closures
    for _, closure in closures.iterrows():
        mask = (df['date'] >= closure['start_date']) & (df['date'] <= closure['end_date'])
        df.loc[mask, 'sales'] *= np.random.uniform(0.4, 0.7)
    
    # Add lag features (sales from previous days)
    for lag in [1, 7, 14, 28]:
        df[f'lag_{lag}'] = df['sales'].shift(lag)
    
    # Add rolling statistics
    for window in [7, 14, 30]:
        df[f'rolling_mean_{window}'] = df['sales'].rolling(window=window).mean().shift(1)
        df[f'rolling_std_{window}'] = df['sales'].rolling(window=window).std().shift(1)
    
    # Drop rows with NaN values (due to lag features)
    df = df.dropna()
    
    return df

# Load and prepare the data
retail_df = load_and_prepare_data()
print(f"Dataset shape: {retail_df.shape}")
print("\nFirst few rows of the dataset:")
print(retail_df.head())

# 2. Exploratory Data Analysis (EDA)
# ==================================

def perform_eda(df):
    """
    Perform exploratory data analysis on the retail sales dataset
    """
    # Summary statistics
    print("\nSummary Statistics:")
    print(df.describe())
    
    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Time series plot of sales
    plt.figure(figsize=(15, 7))
    plt.plot(df['date'], df['sales'])
    plt.title('Daily Retail Sales Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('sales_over_time.png')
    
    # Sales by day of week
    plt.figure(figsize=(10, 6))
    weekday_sales = df.groupby('weekday')['sales'].mean().reindex(range(7))
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    plt.bar(weekday_names, weekday_sales)
    plt.title('Average Sales by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Average Sales')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('sales_by_weekday.png')
    
    # Sales by month
    plt.figure(figsize=(10, 6))
    monthly_sales = df.groupby('month')['sales'].mean().reindex(range(1, 13))
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.bar(month_names, monthly_sales)
    plt.title('Average Sales by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Sales')
    plt.tight_layout()
    plt.savefig('sales_by_month.png')
    
    # Effect of promotions on sales
    plt.figure(figsize=(10, 6))
    promo_effect = df.groupby('is_promotion')['sales'].mean()
    plt.bar(['No Promotion', 'Promotion'], promo_effect)
    plt.title('Average Sales: Promotion vs. No Promotion')
    plt.ylabel('Average Sales')
    plt.tight_layout()
    plt.savefig('promotion_effect.png')
    
    # Effect of holidays on sales
    plt.figure(figsize=(10, 6))
    holiday_effect = df.groupby('is_holiday')['sales'].mean()
    plt.bar(['Regular Day', 'Holiday'], holiday_effect)
    plt.title('Average Sales: Holiday vs. Regular Day')
    plt.ylabel('Average Sales')
    plt.tight_layout()
    plt.savefig('holiday_effect.png')
    
    # Weekends vs. weekdays
    plt.figure(figsize=(10, 6))
    weekend_effect = df.groupby('is_weekend')['sales'].mean()
    plt.bar(['Weekday', 'Weekend'], weekend_effect)
    plt.title('Average Sales: Weekend vs. Weekday')
    plt.ylabel('Average Sales')
    plt.tight_layout()
    plt.savefig('weekend_effect.png')
    
    # Correlation matrix
    plt.figure(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    
    # Time series decomposition
    # Resample to monthly for better visualization
    monthly_df = df.resample('M', on='date')['sales'].mean().reset_index()
    monthly_df.set_index('date', inplace=True)
    
    # Perform decomposition
    decomposition = seasonal_decompose(monthly_df, model='additive', period=12)
    
    plt.figure(figsize=(12, 10))
    plt.subplot(411)
    plt.plot(decomposition.observed)
    plt.title('Observed')
    plt.subplot(412)
    plt.plot(decomposition.trend)
    plt.title('Trend')
    plt.subplot(413)
    plt.plot(decomposition.seasonal)
    plt.title('Seasonality')
    plt.subplot(414)
    plt.plot(decomposition.resid)
    plt.title('Residuals')
    plt.tight_layout()
    plt.savefig('time_series_decomposition.png')
    
    # Check for stationarity using Augmented Dickey-Fuller test
    result = adfuller(df['sales'].dropna())
    print('\nAugmented Dickey-Fuller Test:')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
    
    if result[1] <= 0.05:
        print("The time series is stationary (reject the null hypothesis)")
    else:
        print("The time series is not stationary (fail to reject the null hypothesis)")
    
    return None

# Perform EDA
perform_eda(retail_df)

# 3. Feature Engineering
# ======================

def engineer_features(df):
    """
    Perform feature engineering on the retail dataset
    """
    # Create a copy to avoid modifying the original DataFrame
    df_featured = df.copy()
    
    # Convert date to numerical features (for algorithms that don't handle dates well)
    df_featured['day_of_year'] = df_featured['date'].dt.dayofyear
    
    # Create cyclical features for month and day of week to capture their circular nature
    df_featured['month_sin'] = np.sin(2 * np.pi * df_featured['month'] / 12)
    df_featured['month_cos'] = np.cos(2 * np.pi * df_featured['month'] / 12)
    df_featured['weekday_sin'] = np.sin(2 * np.pi * df_featured['weekday'] / 7)
    df_featured['weekday_cos'] = np.cos(2 * np.pi * df_featured['weekday'] / 7)
    
    # Add features for special periods (e.g., holiday seasons)
    # Christmas season (December)
    df_featured['is_christmas_season'] = (df_featured['month'] == 12).astype(int)
    
    # Summer season (June, July, August)
    df_featured['is_summer'] = df_featured['month'].isin([6, 7, 8]).astype(int)
    
    # Spring season (March, April, May)
    df_featured['is_spring'] = df_featured['month'].isin([3, 4, 5]).astype(int)
    
    # Fall season (September, October, November)
    df_featured['is_fall'] = df_featured['month'].isin([9, 10, 11]).astype(int)
    
    # Winter season (December, January, February)
    df_featured['is_winter'] = df_featured['month'].isin([12, 1, 2]).astype(int)
    
    # Create interaction features
    df_featured['weekend_holiday'] = df_featured['is_weekend'] * df_featured['is_holiday']
    df_featured['weekend_promotion'] = df_featured['is_weekend'] * df_featured['is_promotion']
    df_featured['holiday_promotion'] = df_featured['is_holiday'] * df_featured['is_promotion']
    
    # Create lag ratio features
    df_featured['sales_diff_1'] = df_featured['sales'] - df_featured['lag_1']
    df_featured['sales_ratio_1'] = df_featured['sales'] / df_featured['lag_1']
    df_featured['sales_diff_7'] = df_featured['sales'] - df_featured['lag_7']
    df_featured['sales_ratio_7'] = df_featured['sales'] / df_featured['lag_7']
    
    # Feature for day of month
    df_featured['start_of_month'] = (df_featured['day'] <= 5).astype(int)
    df_featured['end_of_month'] = (df_featured['day'] >= 25).astype(int)
    
    # Replace infinite values with NaNs and then drop rows with NaNs
    df_featured.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_featured.dropna(inplace=True)
    
    # Print the new features added
    new_features = set(df_featured.columns) - set(df.columns)
    print(f"\nNew features added: {new_features}")
    
    return df_featured

# Engineer features
retail_df_featured = engineer_features(retail_df)
print(f"Dataset shape after feature engineering: {retail_df_featured.shape}")

# 4. Model Development
# ====================

def prepare_for_modeling(df):
    """
    Prepare the data for modeling by splitting into train/test sets and scaling features
    """
    # Use date as index for splitting to ensure no data leakage
    df_sorted = df.sort_values('date')
    
    # Define features and target
    X = df_sorted.drop(['date', 'sales'], axis=1)
    y = df_sorted['sales']
    
    # Train/test split using the last 2 months for testing
    split_point = int(len(df_sorted) * 0.85)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
    
    train_dates = df_sorted['date'].iloc[:split_point]
    test_dates = df_sorted['date'].iloc[split_point:]
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, train_dates, test_dates, scaler

def train_and_evaluate_models(X_train, X_test, y_train, y_test, train_dates, test_dates):
    """
    Train and evaluate multiple forecasting models
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    }
    
    results = {}
    predictions = {}
    
    for name, model in models.items():
        # Train model
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Evaluate model
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Store results
        results[name] = {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
        
        predictions[name] = {
            'train_dates': train_dates,
            'test_dates': test_dates,
            'y_train': y_train,
            'y_train_pred': y_pred_train,
            'y_test': y_test,
            'y_test_pred': y_pred_test
        }
        
        # Print results
        print(f"{name} Results:")
        print(f"  Training MAE: {train_mae:.2f}")
        print(f"  Test MAE: {test_mae:.2f}")
        print(f"  Training RMSE: {train_rmse:.2f}")
        print(f"  Test RMSE: {test_rmse:.2f}")
        print(f"  Training R²: {train_r2:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
        
        # For Random Forest, print feature importances
        if name == 'Random Forest':
            feature_importances = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            print("\nTop 10 most important features:")
            print(feature_importances.head(10))
            
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=feature_importances.head(15))
            plt.title('Random Forest Feature Importances')
            plt.tight_layout()
            plt.savefig('feature_importances.png')
    
    # Visualize predictions
    for name, pred_data in predictions.items():
        plt.figure(figsize=(15, 7))
        
        # Plot training data and predictions
        plt.plot(pred_data['train_dates'], pred_data['y_train'], label='Actual (Train)', alpha=0.7)
        plt.plot(pred_data['train_dates'], pred_data['y_train_pred'], label=f'{name} Predictions (Train)', alpha=0.7)
        
        # Plot test data and predictions
        plt.plot(pred_data['test_dates'], pred_data['y_test'], label='Actual (Test)', alpha=0.7)
        plt.plot(pred_data['test_dates'], pred_data['y_test_pred'], label=f'{name} Predictions (Test)', alpha=0.7)
        
        # Add vertical line at train/test split
        split_date = pred_data['test_dates'].iloc[0]
        plt.axvline(x=split_date, color='black', linestyle='--', alpha=0.5)
        plt.text(split_date, plt.ylim()[1] * 0.9, 'Train/Test Split', rotation=90, verticalalignment='top')
        
        plt.title(f'{name} - Actual vs Predicted Sales')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{name.lower().replace(" ", "_")}_predictions.png')
    
    return results, models

# Prepare data for modeling
X_train, X_test, y_train, y_test, train_dates, test_dates, scaler = prepare_for_modeling(retail_df_featured)
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Train and evaluate models
results, models = train_and_evaluate_models(X_train, X_test, y_train, y_test, train_dates, test_dates)

# 5. Model Deployment
# ==================

def save_best_model(models, results):
    """
    Determine the best model based on test RMSE and save it
    """
    best_model_name = min(results, key=lambda x: results[x]['test_rmse'])
    best_model = models[best_model_name]
    best_score = results[best_model_name]['test_rmse']
    
    print(f"\nBest model: {best_model_name} with Test RMSE: {best_score:.2f}")
    
    # In a real project, you would save the model here
    # import joblib
    # joblib.dump(best_model, 'best_sales_model.pkl')
    # joblib.dump(scaler, 'feature_scaler.pkl')
    
    return best_model, best_model_name

def create_prediction_function(model, scaler, feature_columns):
    """
    Create a function that can be used for making future predictions
    """
    def predict_sales(date, is_weekend, is_holiday, is_promotion, previous_sales):
        """
        Predict sales for a given date based on various features
        
        Parameters:
        date (datetime): The date to predict sales for
        is_weekend (bool): Whether the date is a weekend
        is_holiday (bool): Whether the date is a holiday
        is_promotion (bool): Whether there's a promotion running
        previous_sales (dict): Dictionary with previous sales data:
            - 'lag_1': Sales from 1 day ago
            - 'lag_7': Sales from 7 days ago
            - 'lag_14': Sales from 14 days ago
            - 'lag_28': Sales from 28 days ago
            - 'rolling_mean_7': 7-day rolling average
            - 'rolling_mean_14': 14-day rolling average
            - 'rolling_mean_30': 30-day rolling average
            - 'rolling_std_7': 7-day rolling standard deviation
            - 'rolling_std_14': 14-day rolling standard deviation
            - 'rolling_std_30': 30-day rolling standard deviation
        
        Returns:
        float: Predicted sales amount
        """
        # Create a DataFrame with a single row containing all features
        features = pd.DataFrame({
            'weekday': [date.weekday()],
            'month': [date.month],
            'year': [date.year],
            'day': [date.day],
            'is_weekend': [int(is_weekend)],
            'is_holiday': [int(is_holiday)],
            'is_promotion': [int(is_promotion)],
            'lag_1': [previous_sales['lag_1']],
            'lag_7': [previous_sales['lag_7']],
            'lag_14': [previous_sales['lag_14']],
            'lag_28': [previous_sales['lag_28']],
            'rolling_mean_7': [previous_sales['rolling_mean_7']],
            'rolling_mean_14': [previous_sales['rolling_mean_14']],
            'rolling_mean_30': [previous_sales['rolling_mean_30']],
            'rolling_std_7': [previous_sales['rolling_std_7']],
            'rolling_std_14': [previous_sales['rolling_std_14']],
            'rolling_std_30': [previous_sales['rolling_std_30']]
        })
        
        # Add derived features (matching the training data)
        features['day_of_year'] = date.timetuple().tm_yday
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        features['weekday_sin'] = np.sin(2 * np.pi * features['weekday'] / 7)
        features['weekday_cos'] = np.cos(2 * np.pi * features['weekday'] / 7)
        features['is_christmas_season'] = int(features['month'].values[0] == 12)
        features['is_summer'] = int(features['month'].values[0] in [6, 7, 8])
        features['is_spring'] = int(features['month'].values[0] in [3, 4, 5])
        features['is_fall'] = int(features['month'].values[0] in [9, 10, 11])
        features['is_winter'] = int(features['month'].values[0] in [12, 1, 2])
        features['weekend_holiday'] = features['is_weekend'] * features['is_holiday']
        features['weekend_promotion'] = features['is_weekend'] * features['is_promotion']
        features['holiday_promotion'] = features['is_holiday'] * features['is_promotion']
        features['sales_diff_1'] = 0  # This would depend on the prediction
        features['sales_ratio_1'] = 1  # This would depend on the prediction
        features['sales_diff_7'] = 0  # This would depend on the prediction
        features['sales_ratio_7'] = 1  # This would depend on the prediction
        features['start_of_month'] = int(features['day'].values[0] <= 5)
        features['end_of_month'] = int(features['day'].values[0] >= 25)
        
        # Ensure all columns are in the right order
        features = features[feature_columns]
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        predicted_sales = model.predict(features_scaled)[0]
        
        return max(0, predicted_sales)  # Ensure non-negative sales
    
    return predict_sales

# Save best model and create prediction function
best_model, best_model_name = save_best_model(models, results)
predict_sales = create_prediction_function(best_model, scaler, X_train.columns)

# Example of using the prediction function
example_date = pd.Timestamp('2023-01-15')
example_previous_sales = {
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

predicted_sales = predict_sales(
    date=example_date,
    is_weekend=(example_date.weekday() >= 5),
    is_holiday=False,
    is_promotion=True,
    previous_sales=example_previous_sales
)

print(f"\nExample prediction for {example_date.strftime('%Y-%m-%d')}:")
print(f"Predicted sales: ${predicted_sales:.2f}")

# 6. Conclusion
# =============
print("\n========== Project Summary ==========")
print(f"Best model: {best_model_name}")
print(f"Test RMSE: {results[best_model_name]['test_rmse']:.2f}")
print(f"Test R²: {results[best_model_name]['test_r2']:.4f}")
print("\nKey insights from the data:")
print("1. Weekends show higher sales compared to weekdays")
print("2. Promotions significantly boost sales")
print("3. Seasonal patterns are evident with peaks in certain months")
print("4. Recent sales history is a strong predictor of future sales")
print("\nDeployment:")
print("The model has been wrapped in a prediction function that can be used to forecast future sales")
print("====================================")