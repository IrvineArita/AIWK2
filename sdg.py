# data loading
from IPython.display import display

import pandas as pd

try:
    df = pd.read_csv('cleaned_covid_data.csv')
    display(df.head())
    print(df.shape)
except FileNotFoundError:
    print("Error: 'cleaned_covid_data.csv' not found.")
    df = None


#data exploration
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Data Types and Descriptive Statistics
print("Data Types:")
print(df.dtypes)
print("\nDescriptive Statistics for Numerical Features:")
numerical_features = df.select_dtypes(include=['number'])
display(numerical_features.describe())

# 2. Distributions of Key Variables
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.hist(df['total_cases'], bins=20)
plt.title('Distribution of Total Cases')
plt.xlabel('Total Cases')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
plt.hist(df['total_deaths'], bins=20)
plt.title('Distribution of Total Deaths')
plt.xlabel('Total Deaths')
plt.ylabel('Frequency')


plt.subplot(1, 3, 3)
plt.hist(df['total_vaccinations_per_hundred'], bins=20)
plt.title('Distribution of Vaccination Rates')
plt.xlabel('Vaccination Rate (%)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# 3. Missing Values
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
print("\nMissing Values Percentage:")
print(missing_percentage)

# 4. Correlation Analysis
correlation_matrix = numerical_features.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()


#Data preparation
#Subtask:
#Prepare the data for modeling by handling missing values and outliers.
# Handling Missing Values
for col in df.columns:
    if df[col].isnull().sum() / len(df) > 0.9:
        df = df.drop(col, axis=1)
    elif df[col].dtype == 'float64' or df[col].dtype == 'int64':
        df[col] = df[col].fillna(df[col].median())
    elif df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])

# Handling Outliers (using IQR method as an example)
for col in df.select_dtypes(include=['number']):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

# Convert Categorical to Numerical (One-hot encoding for 'continent', label encoding for others)
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    if col == 'continent':
        df = pd.get_dummies(df, columns=[col], prefix=[col])
    else:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# Verify Data Types
print(df.dtypes)
display(df.head())


#data splitting
from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
X = df.drop('total_cases', axis=1)
y = df['total_cases']

# Split data into training and temporary sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=df['continent_Oceania']
)

# Split temporary set into validation and testing sets
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=X_temp['continent_Oceania']
)

#future engineering
import pandas as pd
import numpy as np

def feature_engineer(df):
    # Rolling averages
    df['rolling_cases_7'] = df['new_cases'].rolling(window=7, min_periods=1).mean()
    df['rolling_deaths_7'] = df['new_deaths'].rolling(window=7, min_periods=1).mean()

    # Ratios
    df['new_cases_to_new_cases_smoothed_ratio'] = df['new_cases'] / df['new_cases_smoothed'].replace(0, np.nan)
    df['new_deaths_to_new_deaths_smoothed_ratio'] = df['new_deaths'] / df['new_deaths_smoothed'].replace(0, np.nan)
    df['vaccinations_to_population_ratio'] = df['total_vaccinations_per_hundred'] / (df['population'] / 100)

    # Interaction terms
    df['population_density_vaccination_interaction'] = df['population_density'] * df['total_vaccinations_per_hundred']

    # Differences
    df['daily_case_change'] = df['new_cases'].diff()
    df['daily_death_change'] = df['new_deaths'].diff()
    
    # Fill NaN with 0
    df.fillna(0, inplace=True)
    return df


# Apply feature engineering to each dataset
X_train = feature_engineer(X_train.copy())
X_val = feature_engineer(X_val.copy())
X_test = feature_engineer(X_test.copy())

display(X_train.head())

#model training
from sklearn.ensemble import RandomForestRegressor

# Initialize the model
model = RandomForestRegressor(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = model.predict(X_val)


#model optimization
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the model
rf_model = RandomForestRegressor(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=-1  # Use all available cores
)

# Fit GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Get the best estimator
best_rf_model = grid_search.best_estimator_

# Evaluate the best estimator on the validation set
y_pred_val_optimized = best_rf_model.predict(X_val)


#MODEL VISUALIZATION
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Feature vs. Target Variable Relationships
plt.figure(figsize=(12, 6))
sns.scatterplot(x=X_val['total_vaccinations_per_hundred'], y=y_val, color='skyblue')
plt.xlabel('Total Vaccinations per Hundred')
plt.ylabel('Total Cases')
plt.title('Total Vaccinations per Hundred vs. Total Cases')
plt.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(x=X_val['population_density'], y=y_val, color='lightcoral')
plt.xlabel('Population Density')
plt.ylabel('Total Cases')
plt.title('Population Density vs. Total Cases')
plt.show()

# 2. Model Predictions vs. Actual Values
plt.figure(figsize=(8, 8))
sns.scatterplot(x=y_val, y=y_pred_val_optimized, color='green')
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], color='red', linestyle='--')  # Perfect prediction line
plt.xlabel('Actual Total Cases')
plt.ylabel('Predicted Total Cases')
plt.title('Predicted vs. Actual Total Cases')
plt.show()

# 3. Residual Distribution
residuals = y_val - y_pred_val_optimized
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30, color='orange')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.show()

# 4. Feature Importances
plt.figure(figsize=(10, 6))
importances = best_rf_model.feature_importances_
feature_names = X_train.columns
sns.barplot(x=importances, y=feature_names, color='purple')
plt.xlabel("Feature Importance")
plt.ylabel("Feature Name")
plt.title('Feature Importances')
plt.show()
