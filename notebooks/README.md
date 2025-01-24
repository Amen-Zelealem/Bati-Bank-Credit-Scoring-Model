# Title: Bati Bank Transactions Analysis

## Overview
This notebook provides a comprehensive analysis of a dataset containing financial transactions. The goal is to understand the dataset's structure, summarize its statistics, and explore relationships between features.

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

## Summary Statistics
| Statistic          | Count   | Mean         | Std Dev      | Min          | 25%         | 50%          | 75%         | Max           |
|-------------------|---------|--------------|--------------|--------------|-------------|--------------|-------------|---------------|
| CountryCode       | 95662   | 256.0        | 0.0          | 256.0        | 256.0       | 256.0        | 256.0       | 256.0         |
| Amount            | 95662   | 6717.85      | 123306.80    | -1,000,000   | -50.0       | 1000.0       | 2800.0      | 9,880,000     |
| Value             | 95662   | 9900.58      | 123122.09    | 2.0          | 275.0       | 1000.0       | 5000.0      | 9,880,000     |
| PricingStrategy    | 95662   | 2.26         | 0.73         | 0.0          | 2.0         | 2.0          | 2.0         | 4.0           |
| FraudResult       | 95662   | 0.002        | 0.045        | 0.0          | 0.0         | 0.0          | 0.0         | 1.0           |

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

## Recommendations
- **Handling Outliers**: Utilize robust scaling methods or transformations (e.g., logarithmic scaling) to normalize distributions.

# **Feature Engineering Process**

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

## Conclusion
The dataset offers valuable insights into financial transactions, with specific attributes requiring further investigation for effective modeling. The findings highlight the importance of addressing skewness and outliers to enhance predictive performance.