

#Title: AI for COVID-19 Risk Assessment: A Machine Learning Approach to SDG 3

Introduction: SDG Problem Addressed

This project addresses UN Sustainable Development Goal 3: Good Health and Well-being.
Specifically, we focus on the problem of predicting regions at high risk of experiencing a significant number of active COVID-19 cases.
Accurate risk assessment is crucial for enabling targeted public health interventions, optimizing resource allocation (e.g., hospital beds, medical supplies), and informing policy decisions to mitigate the impact of the pandemic and contribute to global health security.
Machine Learning Approach Used

We employed a supervised learning approach to classify regions into distinct risk levels.
The specific model used is a Random Forest Classifier.
This approach was chosen because our problem involves predicting a categorical outcome (high risk or low risk) based on historical data. The Random Forest is known for its robustness and ability to handle complex relationships in data.
The dataset used is a publicly available COVID-19 dataset containing information on cumulative cases, deaths, recoveries, and population for various regions.
Key preprocessing steps included handling missing values and creating a binary 'risk_level' target variable based on the calculated active cases per capita relative to a defined threshold.

Ethical Considerations
Problem: Predicting the likelihood of a disease outbreak in different regions.
ML Approach: Supervised Learning (Classification). We'll classify regions as "high risk" or "low risk" for outbreaks.
Dataset: You could look for public health datasets, such as reported cases of a specific disease, population data, travel data, or even climate data (some diseases are affected by weather).
Tools: Python, pandas (for data manipulation), scikit-learn (for building the classification model).


EXPLANATION OF THE CODE:

Import Libraries: Imports pandas for data manipulation, train_test_split, RandomForestClassifier, and accuracy_score from sklearn.
DATA LOADINg
Subtask:
Load the cleaned COVID-19 dataset.
Reasoning: Load the cleaned COVID-19 dataset and display the first 5 rows and the shape of the dataframe.

DATA EXPLORATION
Subtask:
Explore the loaded COVID-19 dataset to understand its characteristics and identify potential features for modeling.

Reasoning: Analyze the data types, descriptive statistics, distributions, missing values, and correlations to understand the dataset.

DATA PREPARATION
Subtask:
Prepare the data for modeling by handling missing values and outliers.

Reasoning: Handle missing values, outliers, and convert categorical features to numerical representations.


DATA SPLITTINg
Subtask:
Split the prepared data into training, validation, and testing sets.

Reasoning: Split the data into training, validation, and testing sets, stratifying by continent and using the 'total_cases' column as the target variable.

FEATURE ENGINEERING
Subtask:
Engineer new features from the existing ones in the training, validation, and testing sets.

Reasoning: Create new features based on rolling averages, ratios, and differences, then apply the same transformations to the validation and test sets.

MODEL TRAINING
Subtask:
Train a RandomForestRegressor model to predict 'total_cases'.

Reasoning: Train a RandomForestRegressor model to predict 'total_cases' using the training data and make predictions on the validation set.

MODEL OPTIMIZATION
Subtask:
Optimize the RandomForestRegressor model.

Reasoning: Optimize the RandomForestRegressor model using GridSearchCV.

MODEL EVALUATION
Subtask:
Evaluate the optimized RandomForestRegressor model.

Reasoning: Evaluate the optimized RandomForestRegressor model using MSE and R-squared, and provide an analysis of the model's performance.


DATA VISUALIZATION
Subtask:
Visualize key findings from the data exploration, model training, and evaluation.

Reasoning: Visualize the relationships between important features and the target variable, model performance, error distribution, and feature importances.

Making Predictions: Shows how to use your trained model to predict the risk level for a new data point.
We acknowledge the importance of ethical considerations when applying AI to public health.
Data Privacy: We used an aggregated public dataset, which helps mitigate individual privacy concerns compared to using individual-level patient data.


DATA ANALYSIS AND KEY FINDINGS
The initial dataset cleaned_covid_data.csv contains 6931 rows and 67 columns, encompassing various COVID-19 related metrics and socio-economic indicators.
Several columns, particularly those related to 'excess mortality', have a high percentage of missing values (around 94%).
Numerical features exhibit a wide range of values, with distributions for 'total_cases' and 'total_deaths' showing skewness.
After preprocessing, which involved removing columns with >90% missing data, imputing remaining missing numerical values with the median and categorical values with the mode, and handling outliers using the IQR method, the data was ready for modeling.
Categorical features like 'continent' were one-hot encoded, while others were label encoded.
The data was split into training (70%), validation (15%), and testing (15%) sets, stratified by 'continent_Oceania'.
New features were engineered, including rolling averages of cases and deaths, ratios of cases/deaths to their smoothed counterparts, a vaccination-to-population ratio, an interaction term between population density and vaccination rate, and daily case/death changes.
An optimized RandomForestRegressor model was trained to predict 'total_cases'.
The model evaluation on the validation set yielded a Mean Squared Error (MSE) of approximately 19.26 billion, an R-squared (R2) of 0.99997, and a Mean Absolute Error (MAE) of approximately 23,516. While the R2 is very high, indicating the model explains most of the variance, the high MSE and MAE suggest significant absolute errors in predictions.
Visualizations showed the relationship between features and the target, compared predicted vs. actual values, displayed the distribution of residuals, and highlighted feature importances in the optimized model.