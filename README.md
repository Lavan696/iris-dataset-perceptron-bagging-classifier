# Iris Species Classification using Bagging Perceptron Ensemble

This project focuses on classifying **Iris flower species** using a **BaggingClassifier with Perceptron as the base estimator**.  
It combines **OvR Perceptron classifiers**, **bagging ensemble learning**, and rich **visualization-based analysis** to explore model behavior, feature importance, and generalization.

---

## Project Overview

- **Dataset:** Classic Iris dataset from `sklearn.datasets.load_iris`
- **Task:** Multi-class classification (Setosa, Versicolor, Virginica)
- **Modeling Approach:**
  - One-vs-Rest **Perceptron** classifiers (for understanding each class separately)
  - **BaggingClassifier** with `Perceptron` + `StandardScaler` as base estimator
- **Focus Areas:**
  - Ensemble learning using Perceptrons
  - Extensive visualization of decision boundaries, calibration, learning and validation curves
  - Detailed error analysis via confusion matrices and classification report heatmaps

---

## Dataset Details

- **Classes:**  
  - 0 → Setosa  
  - 1 → Versicolor  
  - 2 → Virginica  
- **Features (all in cm):**
  - Sepal length  
  - Sepal width  
  - Petal length  
  - Petal width  
- **Samples:** 150 (balanced: 50 samples per class)

---

## Modeling Workflow

1. **Data Loading & Exploration**
   - Loaded Iris dataset into a Pandas DataFrame.
   - Inspected `.head()`, `.tail()`, `.info()`, `.describe()` and target distribution.
   - Verified balanced class distribution (equal number of samples per class).

2. **Exploratory Data Analysis (EDA)**
   - **Class Distribution Bar Plot** – shows number of samples per species.
   - **Feature Correlation Heatmap** – visualizes correlation between numerical features.
   - **Feature vs Target Correlation** – bar plot showing association of each feature with the encoded class label.
   - **Pair Plot (Colored by Species)** – visualizes feature pair relationships and natural clusters.

3. **Train-Test Split**
   - Split data into training and test sets with:
     - `test_size = 0.2`
     - `stratify = y` to preserve class distribution.
   - Visualized **Train vs Test Class Distribution** to verify stratification.

4. **OvR Perceptron Classifiers**
   - Built **One-vs-Rest targets**:
     - `y0_train`: Setosa vs Rest  
     - `y1_train`: Versicolor vs Rest  
     - `y2_train`: Virginica vs Rest  
   - Created three Perceptron pipelines:
     - `StandardScaler` → `Perceptron(penalty='l2', alpha=1e-4, random_state=42)`
   - Trained:
     - `perceptron_clf0`, `perceptron_clf1`, `perceptron_clf2`
   - Evaluated each with **5-fold Cross Validation** on the training set.

5. **Bagging Perceptron Ensemble**
   - Defined a base estimator pipeline:
     - `StandardScaler` → `Perceptron(penalty='l2', alpha=1e-4, random_state=42)`
   - Built a **BaggingClassifier**:
     - `estimator = base_perceptron_estimator`
     - `n_estimators = 10`
     - `bootstrap = True`
     - `max_samples = 1.0`, `max_features = 1.0`
     - `n_jobs = -1`, `random_state = 42`
   - Trained on `X_train`, `y_train`
   - Evaluated using **5-fold cross_val_score (accuracy)** on training data.

6. **Feature Importance (from Bagging Perceptron)**
   - Extracted `coef_` from each Perceptron in the ensemble.
   - Took absolute values, summed across classes, and averaged across estimators.
   - Constructed a feature importance Series and plotted a **Feature Importance Bar Chart**.

7. **Learning Curves**
   - Used `learning_curve` with:
     - `train_sizes` from 10% to 100%
     - `cv = 5`, `scoring='accuracy'`, `n_jobs=-1`
   - Plotted **Training vs Cross-Validation Accuracy** vs number of training samples.
   - Analyzed bias/variance behavior of the bagged Perceptron ensemble.

8. **Validation Curves (α – Regularization Strength)**
   - Used `validation_curve` for the `alpha` parameter of the Perceptron in the pipeline:
     - `param_name='estimator__perceptron__alpha'`
     - `param_range = logspace(-5, 2, 10)`
   - Plotted **Training vs Cross-validation Accuracy** vs `alpha` on a log scale.
   - Helped identify good regularization values and diagnose under/overfitting.

## Visualizations

This project includes a rich set of visualizations:

### Data & Distribution
- **Class Distribution Bar Plot** – Dataset balance across three species.
- **Train vs Test Class Distribution** – Verifies stratified split.
- **Feature Correlation Heatmap** – Relationships between features.
- **Feature vs Target Correlation Bar Plot** – Feature association with species.
- **Pair Plot (by species_name)** – Feature cluster separation across species.

### Model & Feature Behavior
- **Feature Importance Bar Plot** (from Bagging Perceptron) – Most influential features for classification.
- **Learning Curves (Bagging Perceptron)** – Training vs Cross-validation accuracy vs train size.
- **Validation Curves (Alpha)** – Performance vs regularization strength.

### Performance & Calibration
- **Precision-Recall Curves (One-vs-Rest)** – For each class using `PrecisionRecallDisplay`.
- **ROC Curves (One-vs-Rest)** – ROC per class with random classifier baseline.
- **Calibration Plots (One-vs-Rest)** – Reliability diagrams per class.
- **Calibration Curve (Aggregated)** – Predicted vs true probability behavior.

### Decision Boundary
- Focused on **two most important features**:
  - `petal length (cm)` and `petal width (cm)`
- Trained a specialized Bagging Perceptron only on these two features.
- Plotted:
  - **Decision Regions** (via meshgrid)
  - **Test points overlaid**, colored by species
- Helps visually understand class separability in the most informative feature space.

### Confusion & Error Analysis
- **Confusion Matrix** – Raw counts of correct/misclassified samples.
- **Normalized Confusion Matrix (Misclassification Heatmap)** – Proportions per true class.
- **Classification Report Heatmap**
- 
---

## Metrics (Summary)

The Bagging Perceptron Classifier was evaluated using:
| Metric                             | Score    |
|------------------------------------|----------|
| Cross-Validation Mean Score        | 92.50%   |
| Test Accuracy                      | 93.33%   |
| Precision (Weighted)               | 93.33%   |
| Recall (Weighted)                  | 93.33%   |
| F1-Score (Weighted)                | 93.33%   |
| ROC-AUC Score (OvR)                | 0.965    |
| Log Loss                           | 1.357    |
| Cohen’s Kappa Score                | 0.90     |
| Matthews Correlation Coefficient   | 0.90     |
| Top-k Accuracy for (k = 2)         | 100%     |

These metrics, combined with the confusion matrices and calibration plots, provide a **comprehensive view of model performance, robustness, and calibration**.

---

## Tech Stack

- **Language:** Python  
- **Libraries:**  
  - NumPy  
  - Pandas  
  - Matplotlib  
  - Seaborn  
- **ML Framework:** scikit-learn  
---

## Model Saving

The trained **Bagging Perceptron model** is saved using **Joblib** for easy reuse and deployment:

`python
import joblib
joblib.dump(bagging_clf, "bagging_perceptron_model.joblib")`

## How to Run

1. **Clone the repository**
`bash
git clone https://github.com/your-username/iris-bagging-perceptron.git`

---

##  Author  

**Lavan Kumar Konda**  
-  2nd Year Student at NIT Andhra Pradesh  
-  Passionate about Data Science, Machine Learning, and AI  
-  [LinkedIn](https://www.linkedin.com/in/lavan-kumar-konda/)
   
