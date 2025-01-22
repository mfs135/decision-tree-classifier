# Decision Tree Classifier for Loan Prediction

## Predicting Loan Case Using Decision Tree

<div align="center">
<img height="300" width="700" src="https://media.geeksforgeeks.org/wp-content/uploads/20221212133528/dcsion.png">
</div>

### Introduction
In this workshop, a **Loan dataset** was used to predict loan approval status using a **Decision Tree Classifier**. The dataset includes features such as:
- Gender
- Marital Status
- Dependents
- Education
- Self Employed
- Applicant’s Income
- Co-applicant’s Income
- Loan Amount Terms
- Credit History
- Property Area
- Loan Amount Log
- Loan Status

The goal was to train a decision tree classifier model and improve its accuracy by selecting the best features.

---

## Methodology

### Steps Followed
1. **Import Libraries**:
   - Libraries like `pandas`, `numpy`, `matplotlib`, and `sklearn` were imported for data manipulation, visualization, and model training.

2. **Generate Dataset**:
   - A unique dataset was generated using:
     ```python
     data = dataset.sample(n=550, random_state=48)
     ```
   - This ensures reproducibility by using a fixed random seed.

3. **Save Dataset**:
   - The generated dataset was saved as `MameFasseSALL-2441202_2.csv`.

4. **Exploratory Data Analysis (EDA)**:
   - Functions like `data.describe()`, `data.size`, `data.ndim`, and `data.shape` were used to explore the dataset.
   - Visualization techniques like histograms and boxplots were employed to analyze the data.

5. **Data Preprocessing**:
   - Categorical data was converted into numerical data using `LabelEncoder`.

6. **Model Training**:
   - A **Decision Tree Classifier** was trained using all features initially.
   - Feature selection was performed to improve model accuracy.

---

## Feature Selection

### Baseline Model (All Features)
- **Accuracy**: 0.63
- **Features Used**: All features except Loan ID.

### Improved Model (Selective Features)
- **Accuracy**: 0.70
- **Features Dropped**: Education, Co-applicant’s Income, Self Employed.
- **Features Selected**:
  - Gender
  - Married
  - Dependents
  - Applicant’s Income
  - Loan Amount Terms
  - Credit History
  - Property Area
  - Loan Amount Log

### Feature Importance Values
| Features               | Score   | Status    |
|------------------------|---------|-----------|
| Gender                 | 0.06469 | Selected  |
| Married                | 0.08967 | Selected  |
| Dependents             | 0.04387 | Selected  |
| Education              | 0.01932 | Dropped   |
| Self Employed          | 0.02104 | Dropped   |
| Applicant’s Income     | 0.21609 | Selected  |
| Co-Applicant’s Income  | 0.00583 | Dropped   |
| Loan Amount Terms      | 0.06889 | Selected  |
| Credit History         | 0.24666 | Selected  |
| Property Area          | 0.11124 | Selected  |
| Loan Amount Log        | 0.11270 | Selected  |

---

## Results

### Classification Report: Baseline Model (All Features)
|          | Precision | Recall | F1-score | Support |
|----------|-----------|--------|----------|---------|
| N        | 0.34      | 0.35   | 0.35     | 51      |
| Y        | 0.74      | 0.73   | 0.74     | 131     |
| **Accuracy**      |           |        | 0.63     | 182     |
| **Macro Avg**     | 0.54      | 0.54   | 0.54     | 182     |
| **Weighted Avg**  | 0.63      | 0.63   | 0.63     | 182     |

### Classification Report: Improved Model (Selective Features)
|          | Precision | Recall | F1-score | Support |
|----------|-----------|--------|----------|---------|
| N        | 0.45      | 0.50   | 0.48     | 30      |
| Y        | 0.81      | 0.78   | 0.79     | 80      |
| **Accuracy**      |           |        | 0.70     | 110     |
| **Macro Avg**     | 0.63      | 0.64   | 0.63     | 110     |
| **Weighted Avg**  | 0.71      | 0.70   | 0.70     | 110     |

---

### Key Findings
- The **selective features model** outperformed the **all-features model**, increasing accuracy from **0.63 to 0.70**.
- Precision and recall for the **Negative class** improved from **0.34 to 0.45** and **0.35 to 0.50**, respectively.
- Precision and recall for the **Positive class** improved from **0.74 to 0.81** and **0.73 to 0.78**, respectively.
- Overall metrics like **macro average** and **weighted average** also showed significant improvement.

---

## Tools Used
- **Python**
- **Pandas** (for data manipulation)
- **NumPy** (for numerical operations)
- **Matplotlib** and **Seaborn** (for visualization)
- **Scikit-learn** (for model training and evaluation)

---

## Setup Instructions

### Prerequisites
- Python 3.x
- Required libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

### Steps to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/decision-tree-classifier.git
   cd decision-tree-classifier


1. install the required libraries:

    ```bash
    pip install -r requirements.txt

### License
This project is licensed under the MIT License. See the LICENSE file for details.

