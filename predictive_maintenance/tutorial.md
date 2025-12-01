# Predictive Maintenance - Equipment Failure Prediction

## Problem Overview

Predictive maintenance aims to predict equipment failures before they occur, enabling proactive maintenance scheduling. This reduces unplanned downtime, extends equipment lifespan, and optimizes maintenance costs. This tutorial demonstrates how to use machine learning to predict equipment failures based on sensor data and operational parameters.

## Dataset Description

**Dataset**: AI4I 2020 Predictive Maintenance Dataset

This synthetic dataset reflects real predictive maintenance scenarios and contains:
- **Operational settings**: Air temperature, process temperature, rotational speed, torque
- **Tool wear**: Measured in minutes of tool usage
- **Machine type**: Different machine variants (H, L, M)
- **Failure modes**: 
  - Tool wear failure (TWF)
  - Heat dissipation failure (HDF)
  - Power failure (PWF)
  - Overstrain failure (OSF)
  - Random failures (RNF)
- **Target**: Machine failure (binary) and failure type (multi-class)

**Dataset Source**: UCI Machine Learning Repository - AI4I 2020 Predictive Maintenance Dataset

## Why This Dataset?

1. **Industry relevance**: Predictive maintenance is a critical Industry 4.0 application
2. **Realistic structure**: Synthetic but based on real-world patterns
3. **Multiple failure modes**: Demonstrates multi-class classification
4. **Temporal aspects**: Tool wear introduces time-dependent patterns
5. **Cost implications**: Demonstrates business value of ML in operations

## Exploratory Data Analysis (EDA) - Rationale

### 1. **Data Overview and Structure**
- **Why**: Understanding the dataset structure helps identify:
  - Temporal patterns (if time-series data)
  - Feature types and ranges
  - Missing values or data quality issues
- **What to look for**: Data shape, feature types, basic statistics, missing values

### 2. **Target Variable Analysis**
- **Why**: Understanding failure patterns is crucial:
  - Failure rate (usually low - imbalanced dataset)
  - Failure type distribution
  - Failure patterns over time or by machine type
- **What to look for**: 
  - Class distribution (likely imbalanced)
  - Failure type frequencies
  - Temporal patterns in failures

### 3. **Operational Parameter Analysis**
- **Why**: Sensor readings and operational settings directly relate to equipment health:
  - Normal operating ranges
  - Relationships between parameters (e.g., temperature and rotational speed)
  - Outliers that may indicate problems
- **What to look for**: 
  - Distributions of temperature, speed, torque
  - Correlations between operational parameters
  - Parameter ranges by machine type

### 4. **Tool Wear Analysis**
- **Why**: Tool wear is a key indicator of equipment condition:
  - Wear progression patterns
  - Relationship between wear and failures
  - Wear thresholds that trigger failures
- **What to look for**: 
  - Tool wear distribution
  - Tool wear vs. failure relationship
  - Wear patterns by machine type

### 5. **Machine Type Analysis**
- **Why**: Different machine types may have different failure patterns:
  - Failure rates by machine type
  - Operational parameter differences
  - Machine-specific failure modes
- **What to look for**: 
  - Failure distribution by machine type
  - Parameter differences across machine types
  - Machine type as a feature

### 6. **Temporal Patterns**
- **Why**: Equipment degradation is time-dependent:
  - Failure patterns over time
  - Tool wear progression
  - Parameter trends before failures
- **What to look for**: 
  - Time series plots of parameters
  - Parameter changes before failures
  - Degradation patterns

### 7. **Correlation Analysis**
- **Why**: Understanding relationships helps in:
  - Feature selection
  - Identifying redundant sensors
  - Understanding failure mechanisms
- **What to look for**: Correlation matrix, highly correlated features

## Feature Engineering - Rationale

### 1. **Handling Missing Values**
- **Why**: Sensor data may have missing values due to sensor failures or data collection issues
- **Strategy**: 
  - Forward fill for time-series data (carry last known value)
  - Interpolation for continuous sensors
  - Median imputation for non-temporal data

### 2. **Creating Time-Based Features**
- **Why**: Equipment degradation is time-dependent. Temporal features capture:
  - Wear progression
  - Degradation trends
  - Time since last maintenance
- **Strategy**: 
  - Time since start/installation
  - Time since last failure
  - Cyclical time features (if applicable)

### 3. **Rolling Statistics**
- **Why**: Recent trends in sensor readings often predict failures better than point values:
  - Moving averages capture trends
  - Rolling standard deviations capture variability
  - Recent changes indicate degradation
- **Strategy**: 
  - Rolling mean (e.g., last 10, 50, 100 observations)
  - Rolling standard deviation
  - Rolling min/max
  - Rate of change (derivative)

### 4. **Lag Features**
- **Why**: Previous sensor readings can predict future failures:
  - Captures temporal dependencies
  - Identifies gradual degradation
- **Strategy**: 
  - Lag features (previous 1, 2, 3... time steps)
  - Differences between current and lagged values

### 5. **Feature Interactions**
- **Why**: Failure often results from interactions between parameters:
  - Temperature × Rotational speed (heat generation)
  - Torque × Speed (power)
  - Tool wear × Operating conditions
- **Strategy**: 
  - Multiply related features
  - Create ratio features (e.g., temperature/speed)
  - Polynomial features for non-linear relationships

### 6. **Threshold-Based Features**
- **Why**: Operating outside normal ranges increases failure risk:
  - High/low temperature flags
  - Speed/torque limit violations
  - Tool wear thresholds
- **Strategy**: 
  - Binary features for threshold violations
  - Distance from normal operating range
  - Z-scores for parameter deviations

### 7. **Machine Type Encoding**
- **Why**: Different machine types have different characteristics
- **Strategy**: 
  - One-hot encoding for machine type
  - Or create machine-specific features

### 8. **Feature Scaling**
- **Why**: Sensor readings have different scales and units
- **Strategy**: 
  - StandardScaler for distance-based algorithms
  - RobustScaler if outliers are important to preserve
  - MinMaxScaler for neural networks

### 9. **Outlier Treatment**
- **Why**: Outliers in sensor data can be:
  - Sensor errors (should be handled)
  - Actual anomalies (may be failure indicators - should be preserved)
- **Strategy**: 
  - Identify using domain knowledge and statistical methods
  - Cap extreme values rather than removing (may indicate problems)
  - Use robust scaling methods

## Machine Learning Approach

### Model Selection Rationale

1. **Logistic Regression**
   - **Why**: 
     - Baseline model
     - Interpretable (coefficients show feature importance)
     - Fast training
   - **Use case**: Baseline and when interpretability is needed

2. **Random Forest**
   - **Why**: 
     - Handles non-linear relationships well
     - Feature importance scores
     - Robust to outliers
     - Good default for tabular data
   - **Use case**: When you want good performance with minimal tuning

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

- **Why**: Equipment failures are rare events (imbalanced dataset)
- **Strategies**:
  - **SMOTE**: Synthetic oversampling of minority class
  - **Class weights**: Penalize misclassifying failures more
  - **Threshold tuning**: Adjust decision threshold to optimize for recall
  - **Ensemble methods**: Use methods that handle imbalance well

### Evaluation Metrics

- **Accuracy**: Can be misleading with imbalanced data
- **Precision**: Of predicted failures, how many are actual failures?
- **Recall (Sensitivity)**: Of actual failures, how many did we predict? **CRITICAL** - missing failures is costly
- **F1-Score**: Balances precision and recall
- **ROC-AUC**: Model's ability to distinguish failures
- **Precision-Recall AUC**: Better for imbalanced datasets
- **Confusion Matrix**: Detailed breakdown

**Why these metrics?**: In predictive maintenance:
- **High recall is critical**: Missing a failure (false negative) causes unplanned downtime
- **Precision matters**: Too many false alarms (false positives) reduce trust in the system
- **Cost-benefit**: Balance between maintenance costs and downtime costs

## Expected Outcomes

After completing this tutorial, you should be able to:
1. Load and explore predictive maintenance sensor data
2. Perform EDA to understand equipment behavior and failure patterns
3. Engineer temporal and interaction features for time-dependent problems
4. Handle class imbalance common in failure prediction
5. Build and compare ML models using scikit-learn
6. Evaluate models using appropriate metrics for imbalanced classification
7. Interpret results in the context of maintenance scheduling

## Key Takeaways

- **Temporal features matter**: Equipment degradation is time-dependent - capture this in features
- **Class imbalance is common**: Failure prediction is inherently imbalanced - use appropriate techniques
- **Feature engineering is critical**: Rolling statistics and interactions often improve performance significantly
- **Recall is often more important**: Missing failures is usually costlier than false alarms
- **Domain knowledge helps**: Understanding equipment behavior guides feature engineering
- **Iterative improvement**: Start simple, add complexity based on results

