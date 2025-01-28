# Bati-Bank-Credit-Scoring-Model
Credit Scoring Model for Bati Bank's Buy-Now-Pay-Later Service. This project aims to develop a robust credit scoring system by defining risk categories, selecting predictive features, and creating models to estimate risk probabilities, credit scores, and optimal loan terms based on data from a partnering eCommerce platform.

## Project Overview EDA Analysis
This Task-1 EDA Analysis includes a comprehensive analysis of financial transactions to inform the credit scoring model. The objectives are to understand the dataset's structure, summarize its statistics, and explore relationships between features.

## Completed Tasks

1. **Overview of the Data**: 
   - Analyzed the structure, including the number of rows, columns, and data types.

2. **Summary Statistics**: 
   - Generated summary statistics to understand central tendency, dispersion, and shape of distributions.

3. **Distribution of Numerical Features**: 
   - Visualized distributions to identify patterns, skewness, and potential outliers.

4. **Distribution of Categorical Features**: 
   - Analyzed categorical features for insights into frequency and variability.

5. **Correlation Analysis**: 
   - Investigated relationships between numerical features.

6. **Identifying Missing Values**: 
   - Checked for missing values to determine data completeness.

7. **Outlier Detection**: 
   - Used box plots to identify outliers.

## Summary of the Dataset
- **Total Transactions**: 95,662
- **Attributes**: 15
  - **Categorical Identifiers**: BatchId, AccountId, CustomerId
  - **Financial Metrics**: Amount, Value
  - **Timestamps**: TransactionStartTime

**Missing Values**: None

## Key Insights
- **Right-Skewness**: Most numerical features are right-skewed, indicating a few extreme values significantly influence the mean.
- **Prominent Peaks**: Key variables like `CountryCode`, `Amount`, and `Value` show distinct peaks.
- **Preferred Pricing Strategy**: A strong preference for pricing strategy 2.
- **Majority Non-Fraudulent Transactions**: The dataset shows a low incidence of fraud.

## Insights from Categorical Features
- **CurrencyCode**: All transactions are in UGX (Ugandan Shilling).
- **ProviderId**: Dominated by two providers.
- **ProductCategory**: Primarily consists of airtime and financial services.
- **ChannelId**: Most transactions occur via two preferred channels.

## Correlation Analysis
- **Amount and Value**: Strong positive correlation; consider removing `Value` for modeling.
- **PricingStrategy**: No significant correlation with other features.

## Outlier Analysis
- Significant outliers identified in `Amount` and `Value`, warranting further investigation.

## **Feature Engineering Process**

The feature engineering process involves several key steps to prepare the dataset for analysis and model training:

## **Steps Involved**

1. **Encoding Categorical Variables**:
   - Categorical variables were encoded using one-hot encoding to convert them into a numerical format suitable for machine learning algorithms.

2. **Standardizing Numerical Features**:
   - Numerical features were standardized using the `StandardScaler`. This ensures consistency in scale across features, which is crucial for many machine learning models.

3. **Handling Missing Values**:
   - During the feature engineering process, the new feature `Std_Transaction_Amount` was found to have **712 missing values**. To ensure data completeness, these missing values were imputed with the mean of the feature.

## **Summary**

These steps improve the quality of the dataset, making it more suitable for further analysis and predictive modeling. Proper encoding, scaling, and handling of missing values are essential for building effective machine learning models.

## **Modeling**

### **Model Selection and Training**

### **Split the Data**
Divide the dataset into training and testing sets to evaluate model performance on unseen data.

### **Choose Models**
Select at least two models from the following options:
- Logistic Regression
- Decision Trees
- Random Forest
- Gradient Boosting Machines (GBM)

### **Train the Models**
Train the selected models on the training dataset.

### **Hyperparameter Tuning**
Enhance model performance through hyperparameter tuning using techniques like:
- Grid Search
- Random Search

## **Model Evaluation**
Assess model performance using metrics such as:
- **Accuracy**: Ratio of correctly predicted observations to total observations.
- **Precision**: Ratio of correctly predicted positive observations to total predicted positives.
- **Recall (Sensitivity)**: Ratio of correctly predicted positive observations to all actual positives.
- **F1 Score**: Weighted average of Precision and Recall.
- **ROC-AUC**: Area Under the Receiver Operating Characteristic Curve, measuring the model's ability to distinguish between classes.


# Folder Structure
```
+---.github
| └── workflows
| └── blank.yml
+---.vscode
| └── settings.json
+---notebooks
| ├── init.ipynb
| ├── eda_analysis.ipynb
| ├── feature_engineering.ipynb
| └── README.md
+---scripts
| ├── init.py
| ├── data_loader.py
| ├── eda_analysis.py
| ├── feature_engineering.py
| └── README.md
+---src
| └── README.md
| └── init.py
+---tests
| └── init.py
| ├── README.md
| └── test_data_loader.py
| └── test_eda_analysis.py
| └── test_feature_engineering.py
| ├── .gitignore
| ├── LICENSE
| ├── README.md
| └── requirements.txt
```