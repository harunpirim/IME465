# Introduction to Machine Learning - Industry Applications

This repository contains comprehensive tutorials and Jupyter notebooks for four real-world machine learning problems across different industries. Each tutorial demonstrates the complete ML workflow: data exploration, feature engineering, model building, and evaluation using **scikit-learn**.

## Repository Structure

```
IME465/
├── quality_control/
│   ├── tutorial.md
│   └── quality_control.ipynb
├── healthcare/
│   ├── tutorial.md
│   └── healthcare.ipynb
├── predictive_maintenance/
│   ├── tutorial.md
│   └── predictive_maintenance.ipynb
├── supply_chain/
│   ├── tutorial.md
│   └── supply_chain.ipynb
└── README.md
```

## Problems Covered

### 1. Quality Control - Manufacturing Defect Prediction
**Location**: `quality_control/`

Predict manufacturing defects in steel plates using geometric and material properties.

- **Dataset**: Steel Plates Faults Dataset (UCI ML Repository)
- **Problem Type**: Binary Classification
- **Key Concepts**: 
  - Feature engineering for geometric properties
  - Handling correlated features
  - Model comparison and evaluation

### 2. Healthcare - Heart Disease Prediction
**Location**: `healthcare/`

Predict heart disease presence based on patient demographics and medical measurements.

- **Dataset**: Heart Disease UCI Dataset
- **Problem Type**: Binary Classification
- **Key Concepts**:
  - Clinical feature engineering
  - Age group categorization
  - Medical threshold features
  - Interpretability in healthcare

### 3. Predictive Maintenance - Equipment Failure Prediction
**Location**: `predictive_maintenance/`

Predict equipment failures before they occur using sensor data and operational parameters.

- **Dataset**: AI4I 2020 Predictive Maintenance Dataset
- **Problem Type**: Binary Classification (Imbalanced)
- **Key Concepts**:
  - Temporal feature engineering
  - Handling class imbalance
  - Interaction features
  - Recall-focused evaluation

### 4. Supply Chain - Delivery Delay Prediction
**Location**: `supply_chain/`

Predict delivery delays based on order characteristics, shipping details, and temporal patterns.

- **Dataset**: Supply Chain Delivery Dataset (synthetic, based on real patterns)
- **Problem Type**: Binary Classification
- **Key Concepts**:
  - Temporal feature engineering
  - Geographic features
  - Categorical encoding
  - Cyclical time encoding

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Jupyter Notebook or JupyterLab
- Required Python packages:
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn jupyter
  ```

### Installation

1. Clone or download this repository
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
   (If requirements.txt doesn't exist, install packages listed above)

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. Navigate to any problem folder and open the `.ipynb` file

## How to Use This Repository

### For Students

1. **Start with the Tutorial**: Read the `tutorial.md` file in each folder to understand:
   - The problem domain and business context
   - Why specific EDA techniques are chosen
   - Rationale behind feature engineering decisions
   - Model selection reasoning

2. **Follow the Notebook**: Work through the `.ipynb` file step by step:
   - Run each cell
   - Understand the code and outputs
   - Experiment with modifications
   - Answer the questions and exercises

3. **Compare Approaches**: Notice how different problems require different:
   - EDA techniques
   - Feature engineering strategies
   - Model choices
   - Evaluation metrics

### For Instructors

- Each tutorial includes detailed explanations of:
  - **Why** specific techniques are chosen (not just what)
  - Domain-specific considerations
  - Trade-offs in model selection
  - Business impact of predictions

- Use these as:
  - Lecture examples
  - Lab assignments
  - Project templates
  - Discussion starters

## Key Learning Objectives

After completing these tutorials, you should be able to:

1. **Perform Domain-Specific EDA**
   - Understand data in context
   - Identify relevant patterns
   - Detect data quality issues

2. **Engineer Meaningful Features**
   - Create domain-informed features
   - Handle different data types
   - Transform features appropriately

3. **Build and Evaluate ML Models**
   - Select appropriate algorithms
   - Handle class imbalance
   - Choose relevant evaluation metrics
   - Interpret results in business context

4. **Apply ML to Real Problems**
   - Understand business requirements
   - Balance model complexity and interpretability
   - Consider deployment constraints

## Common Workflow Across All Problems

1. **Data Loading & Initial Exploration**
   - Load dataset
   - Check basic statistics
   - Identify missing values
   - Understand data types

2. **Exploratory Data Analysis (EDA)**
   - Target variable distribution
   - Feature distributions
   - Correlation analysis
   - Feature-target relationships

3. **Feature Engineering**
   - Handle missing values
   - Create derived features
   - Encode categorical variables
   - Scale/normalize features

4. **Model Building**
   - Split data (train/test)
   - Train multiple models
   - Compare performance
   - Select best model

5. **Evaluation & Interpretation**
   - Calculate metrics
   - Visualize results
   - Interpret in domain context
   - Discuss business impact

## Dataset Sources

- **Quality Control**: UCI ML Repository - Steel Plates Faults Dataset
- **Healthcare**: UCI ML Repository - Heart Disease Dataset
- **Predictive Maintenance**: UCI ML Repository - AI4I 2020 Predictive Maintenance Dataset
- **Supply Chain**: Synthetic dataset based on real-world supply chain patterns

**Note**: The notebooks use synthetic data generators that mimic the real datasets. For production use, download the actual datasets from the UCI ML Repository or other sources.

## Technologies Used

- **Python**: Programming language
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib & seaborn**: Data visualization
- **scikit-learn**: Machine learning algorithms and utilities

## Evaluation Metrics Explained

Different problems emphasize different metrics:

- **Accuracy**: Overall correctness (can be misleading with imbalanced data)
- **Precision**: Of positive predictions, how many are correct?
- **Recall**: Of actual positives, how many did we catch?
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Model's ability to distinguish classes

**Why it matters**:
- **Quality Control**: Balance precision and recall (both false positives and false negatives are costly)
- **Healthcare**: High recall is critical (missing disease is dangerous)
- **Predictive Maintenance**: High recall is critical (missing failures causes downtime)
- **Supply Chain**: Balance precision and recall (customer satisfaction vs. false alarms)

## Contributing

This is an educational repository. Suggestions for improvements are welcome:
- Additional problems/domains
- Better explanations
- Code improvements
- Additional exercises

## License

This repository is intended for educational purposes.

## Acknowledgments

- UCI Machine Learning Repository for providing datasets
- scikit-learn developers for excellent ML tools
- The open-source community for data science tools

## Contact

For questions or issues, please refer to the tutorial files in each problem folder or consult the course instructor.

---

**Happy Learning!** 🚀

