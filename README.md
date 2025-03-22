# Credit Card Fraud Detection System
This project implements a machine learning-based system to detect fraudulent credit card transactions using ensemble methods with advanced feature engineering.
Overview
The system uses a voting ensemble of multiple machine learning algorithms (Logistic Regression, Random Forest, Gradient Boosting, and XGBoost) to classify transactions as legitimate or fraudulent. The model handles class imbalance using SMOTE oversampling and employs comprehensive feature engineering to improve detection accuracy.
Dataset


## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Output Files and Visualization](#output-files-and-visualization)
- [Model Performance](#model-performance)
- [Contributing](#contributing)
- [License](#license)

## Overview

The system uses a voting ensemble of multiple machine learning algorithms (Logistic Regression, Random Forest, Gradient Boosting, and XGBoost) to classify transactions as legitimate or fraudulent. The model handles class imbalance using SMOTE oversampling and employs comprehensive feature engineering to improve detection accuracy.

## Dataset

This project uses the Credit Card Fraud Detection dataset available on Kaggle:
- Dataset URL: https://www.kaggle.com/datasets/kartik2112/fraud-detection

### Download Instructions:
1. Visit the Kaggle URL above
2. Click the "Download" button (you may need to sign in to Kaggle)
3. Extract the downloaded ZIP file
4. Place the `fraudTrain.csv` and `fraudTest.csv` files in the same directory as this script

## Installation

### Clone the repository
```bash
git clone https://github.com/gkmohith17/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

### Install dependencies
The project includes a `requirements.txt` file with all necessary dependencies:

```bash
pip install -r requirements.txt
```

This will install all required packages including pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, imbalanced-learn, and joblib.

## Usage

1. Download and place the dataset files as described in the [Dataset](#dataset) section
2. Run the main script:

```bash
python credit_card_fraud_detection.py
```

### Console Output

During execution, the script will display:
- Dataset loading information
- Data exploration statistics
- Training progress information
- Model evaluation metrics
- Misclassification analysis

Example output:
```
=== Credit Card Fraud Detection System ===
Loading datasets...
Train data: 555719 rows, Test data: 555719 rows

--- Data Exploration ---
Legitimate: 551294, Fraudulent: 4425 (0.80%)

No missing values found
Training set: 555719 samples, Test set: 555719 samples
Training ensemble model...

--- Ensemble Model Evaluation ---
              precision    recall  f1-score   support
           0       1.00      0.99      1.00    551294
           1       0.85      0.92      0.88      4425
    accuracy                           0.99    555719
   macro avg       0.93      0.96      0.94    555719
weighted avg       0.99      0.99      0.99    555719

ROC AUC: 0.9879, PR-AUC: 0.8912
False Positive Rate: 0.0054, False Negative Rate: 0.0831

False Positives: 2977, False Negatives: 368
FP Avg: Amount $1102.47, Age 42.3 yrs
FN Avg: Amount $2865.79, Age 39.5 yrs

Model saved to fraud_detection_model.pkl

=== Fraud Detection System Complete ===
Model achieved ROC-AUC: 0.9879 and PR-AUC: 0.8912
```

## Features

The system implements several advanced features:
- **Temporal Features**: Time-based patterns including hour of day, day of week, weekend detection
- **Geographic Analysis**: Distance calculation between merchant and customer
- **Transaction Patterns**: Time between transactions, transaction velocity
- **Customer Profiling**: Age-based features, customer transaction averages
- **Merchant Risk Assessment**: Merchant and category fraud rates

## Output Files and Visualization

### Model File
- `fraud_detection_model.pkl`: The trained model and scaler saved for deployment or later use
  - This file can be loaded using joblib: `model_data = joblib.load('fraud_detection_model.pkl')`

### Visualization
- `feature_importance.png`: A bar chart showing the most important features for fraud detection
  - Open this file with any image viewer to understand which features contribute most to the model's decisions

Example of how to view these files:

```python
# To load the saved model
import joblib
model_data = joblib.load('fraud_detection_model.pkl')
model = model_data['model']
scaler = model_data['scaler']

# Make predictions with loaded model
predictions = model.predict(scaler.transform(new_data))
```

## Model Performance

The model's performance is evaluated using:
- ROC-AUC score
- Precision-Recall AUC (PR-AUC)
- Classification report (precision, recall, F1-score)
- Confusion matrix analysis
- False positive and false negative analysis

## Contributing

Contributions are welcome! Here's how you can contribute to this project:

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Implement your changes**:
   - Add new features
   - Fix bugs
   - Improve documentation
   - Enhance model performance

4. **Add tests** for any new functionality

5. **Ensure code quality**:
   - Follow PEP 8 style guidelines
   - Add appropriate comments
   - Update documentation

6. **Commit your changes**:
   ```bash
   git commit -m "Add detailed description of your changes"
   ```

7. **Push to your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Create a Pull Request**

### Contribution Ideas
- Implement additional ML algorithms
- Add more feature engineering techniques
- Create interactive visualizations
- Add real-time prediction capabilities
- Improve model explainability

## License

[MIT License](LICENSE)

## Author

[Your Name]

---

If you find this project useful, please give it a star ⭐ on GitHub!