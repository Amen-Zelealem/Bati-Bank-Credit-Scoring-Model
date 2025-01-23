# Bati-Bank-Credit-Scoring-Model
Credit Scoring Model for Bati Bank's Buy-Now-Pay-Later Service. This project aims to develop a robust credit scoring system by defining risk categories, selecting predictive features, and creating models to estimate risk probabilities, credit scores, and optimal loan terms based on data from a partnering eCommerce platform.

## Task-1 Project Overview EDA Analysis
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


# Folder Structure
```
+---.github
| └── workflows
| └── blank.yml
+---.vscode
| └── settings.json
+---notebooks
| ├── init.ipynb
| └── README.md
+---scripts
| ├── init.py
| └── README.md
+---src
| └── README.md
| └── init.py
+---tests
| ├── README.md
| └── init.py
| ├── .gitignore
| ├── README.md
| └── requirements.txt
```

# Conclusion
The dataset offers valuable insights into financial transactions, which will inform the credit scoring model for Bati Bank. Addressing skewness and outliers will enhance predictive performance and model reliability.