# Supply Chain - Delivery Delay Prediction

## Problem Overview

Supply chain management involves coordinating the flow of goods from suppliers to customers. Predicting delivery delays helps companies:
- Improve customer satisfaction through proactive communication
- Optimize inventory levels
- Reduce costs from expedited shipping
- Better plan production schedules

This tutorial demonstrates how to use machine learning to predict delivery delays based on order characteristics, shipping details, and historical patterns.

## Dataset Description

**Dataset**: Supply Chain Delivery Dataset

This dataset contains information about orders and their delivery status:
- **Order characteristics**: Order quantity, product category, product value
- **Shipping details**: Shipping mode, carrier, origin and destination locations
- **Temporal features**: Order date, shipping date, scheduled delivery date
- **Geographic features**: Origin and destination cities, distances
- **Historical patterns**: Customer order history, supplier performance
- **External factors**: Weather conditions, holiday periods

**Target Variable**: Binary classification - delivery delayed (1) or on-time (0)

**Note**: This tutorial uses a synthetic dataset that reflects real-world supply chain scenarios. In practice, you would use actual company data.

## Why This Dataset?

1. **Business impact**: Delivery delays directly affect customer satisfaction and costs
2. **Multi-factor problem**: Demonstrates handling multiple feature types (categorical, numerical, temporal)
3. **Real-world complexity**: Includes realistic challenges like missing data, imbalanced classes
4. **Actionable insights**: Predictions can directly inform operational decisions

## Exploratory Data Analysis (EDA) - Rationale

### 1. **Data Overview and Quality**
- **Why**: Supply chain data often comes from multiple systems and may have quality issues:
  - Missing values from incomplete records
  - Inconsistent date formats
  - Data entry errors
- **What to look for**: 
  - Missing value patterns
  - Data types and formats
  - Inconsistencies in categorical values

### 2. **Target Variable Distribution**
- **Why**: Understanding delay patterns:
  - Overall delay rate
  - Class imbalance (delays may be rare or common)
  - Temporal patterns in delays
- **What to look for**: 
  - Class distribution
  - Delay rate over time
  - Seasonal patterns

### 3. **Temporal Analysis**
- **Why**: Delivery delays often have temporal patterns:
  - Seasonal effects (holidays, weather)
  - Day of week effects (weekends, Mondays)
  - Month/quarter effects
  - Trends over time
- **What to look for**: 
  - Delay rates by day of week, month, season
  - Time series of delay rates
  - Holiday effects

### 4. **Geographic Analysis**
- **Why**: Location affects delivery times:
  - Distance impacts delivery time
  - Regional differences in infrastructure
  - Urban vs. rural delivery challenges
- **What to look for**: 
  - Delay rates by origin/destination
  - Distance distributions
  - Geographic patterns

### 5. **Shipping Mode Analysis**
- **Why**: Different shipping methods have different delay risks:
  - Express vs. standard shipping
  - Carrier performance differences
  - Mode-specific challenges
- **What to look for**: 
  - Delay rates by shipping mode
  - Delay rates by carrier
  - Mode × distance interactions

### 6. **Product and Order Characteristics**
- **Why**: Order attributes affect delivery:
  - Product category (size, fragility)
  - Order quantity (bulk orders may take longer)
  - Order value (high-value orders may get priority)
- **What to look for**: 
  - Delay rates by product category
  - Order quantity/value distributions
  - Relationships with delays

### 7. **Correlation Analysis**
- **Why**: Understanding feature relationships:
  - Identify redundant features
  - Discover interaction opportunities
  - Understand multicollinearity
- **What to look for**: Correlation matrix, highly correlated features

## Feature Engineering - Rationale

### 1. **Temporal Feature Engineering**
- **Why**: Time-based patterns are crucial for delay prediction:
  - Day of week (weekends, Mondays often have delays)
  - Month/season (holidays, weather)
  - Days until delivery (urgency)
  - Time since order (processing time)
- **Strategy**: 
  - Extract day of week, month, quarter, year
  - Create binary features for holidays
  - Calculate time differences (order to ship, ship to delivery)
  - Cyclical encoding for periodic patterns (sine/cosine for day of week)

### 2. **Geographic Feature Engineering**
- **Why**: Location significantly impacts delivery:
  - Distance is a key factor
  - Regional characteristics matter
- **Strategy**: 
  - Calculate distance between origin and destination
  - Create region/country features
  - Urban vs. rural indicators
  - Geographic clusters

### 3. **Categorical Encoding**
- **Why**: Many supply chain features are categorical:
  - Shipping mode, carrier, product category
  - Origin/destination locations
- **Strategy**: 
  - One-hot encoding for nominal categories
  - Target encoding for high-cardinality categories (carrier, location)
  - Frequency encoding (how often does this category appear)
  - Ordinal encoding if order matters

### 4. **Aggregate Features**
- **Why**: Historical patterns predict future behavior:
  - Carrier performance history
  - Route performance history
  - Customer delay history
  - Supplier performance
- **Strategy**: 
  - Calculate historical delay rates by carrier, route, customer
  - Rolling averages of past performance
  - Count of previous delays

### 5. **Interaction Features**
- **Why**: Combinations of features often matter:
  - Distance × Shipping mode (long distance + standard shipping = higher delay risk)
  - Quantity × Product category (bulk fragile items)
  - Carrier × Route (some carriers perform better on certain routes)
- **Strategy**: 
  - Multiply or divide related features
  - Create categorical interactions
  - Polynomial features for non-linear relationships

### 6. **Derived Features**
- **Why**: Calculated features can capture important relationships:
  - Order value per unit (value/quantity)
  - Processing time (order to ship)
  - Lead time (order to delivery)
  - Urgency indicators
- **Strategy**: 
  - Create ratio features
  - Calculate time differences
  - Create urgency scores

### 7. **Handling Missing Values**
- **Why**: Supply chain data often has missing values:
  - Incomplete records
  - Optional fields
  - Data collection issues
- **Strategy**: 
  - For numerical: median or mean imputation
  - For categorical: mode or "unknown" category
  - Consider if missingness is informative (create "is_missing" features)

### 8. **Feature Scaling**
- **Why**: Features have different scales:
  - Distance (miles/kilometers)
  - Order value (dollars)
  - Quantity (units)
- **Strategy**: 
  - StandardScaler for distance-based algorithms
  - MinMaxScaler if you need bounded ranges
  - RobustScaler if outliers are important

### 9. **Outlier Treatment**
- **Why**: Outliers in supply chain data can be:
  - Data errors (should be corrected)
  - Valid extreme cases (should be handled carefully)
- **Strategy**: 
  - Use domain knowledge (e.g., maximum reasonable distance)
  - Cap extreme values
  - Use robust scaling methods

## Machine Learning Approach

### Model Selection Rationale

1. **Logistic Regression**
   - **Why**: 
     - Baseline model
     - Interpretable (coefficients show feature importance)
     - Fast training
     - Works well with engineered features
   - **Use case**: Baseline and when interpretability is needed

2. **Random Forest**
   - **Why**: 
     - Handles mixed feature types well
     - Captures non-linear relationships
     - Feature importance scores
     - Robust to outliers
   - **Use case**: Good default choice for tabular data

3. **Gradient Boosting**
   - **Why**: 
     - Often best performance
     - Handles complex patterns
     - Feature importance available
   - **Use case**: When maximum accuracy is the goal

4. **Support Vector Machine (SVM)**
   - **Why**: 
     - Effective with many features
     - Can capture complex boundaries
   - **Use case**: When you have many engineered features

### Handling Class Imbalance

- **Why**: Delays may be rare or common depending on the business
- **Strategies**:
  - **SMOTE**: If delays are rare
  - **Class weights**: Adjust model to penalize missing delays
  - **Threshold tuning**: Optimize decision threshold
  - **Cost-sensitive learning**: Weight errors by business cost

### Evaluation Metrics

- **Accuracy**: Overall correctness (may be misleading if imbalanced)
- **Precision**: Of predicted delays, how many are actual delays?
- **Recall**: Of actual delays, how many did we predict? **Important** - missing delays affects customers
- **F1-Score**: Balances precision and recall
- **ROC-AUC**: Model's ability to distinguish delays
- **Precision-Recall AUC**: Better for imbalanced datasets
- **Confusion Matrix**: Detailed breakdown

**Why these metrics?**: In supply chain:
- **Recall matters**: Missing delays (false negatives) leads to customer dissatisfaction
- **Precision matters**: Too many false alarms (false positives) reduces trust and wastes resources
- **Business context**: Consider costs of false positives vs. false negatives

## Expected Outcomes

After completing this tutorial, you should be able to:
1. Load and explore supply chain delivery data
2. Perform comprehensive EDA including temporal and geographic analysis
3. Engineer temporal, geographic, and aggregate features
4. Handle categorical variables with appropriate encoding
5. Build and compare ML models using scikit-learn
6. Evaluate models using appropriate metrics
7. Interpret results in supply chain context

## Key Takeaways

- **Temporal features are crucial**: Time-based patterns strongly predict delays
- **Geographic features matter**: Location and distance significantly impact delivery
- **Historical patterns help**: Aggregate features from past data improve predictions
- **Feature engineering is key**: Well-engineered features often outperform algorithm selection
- **Business context matters**: Choose metrics and thresholds based on business costs
- **Categorical encoding matters**: Choose encoding method based on cardinality and relationships
- **Iterative process**: Start with domain knowledge, refine based on results

