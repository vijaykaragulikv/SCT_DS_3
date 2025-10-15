# Internship Project: Bank Marketing Prediction (Decision Tree Classifier)

## 🎯 Project Goal

This project was completed as an internship task focusing on **predictive modeling** and **hyperparameter tuning**.

The primary objective was to build and optimize a **Decision Tree Classifier** to predict whether a banking customer would subscribe to a term deposit, using their demographic and behavioral data from the UCI Bank Marketing dataset.

## 🛠️ Technology & Libraries Used

| Tool/Library | Purpose |
| :--- | :--- |
| **Python** | Primary programming language |
| **Pandas** | Data loading, cleaning, and manipulation (e.g., One-Hot Encoding) |
| **NumPy** | Numerical operations |
| **Scikit-learn (sklearn)** | Model building, training (`DecisionTreeClassifier`), evaluation, and hyperparameter tuning (`GridSearchCV`) |
| **Matplotlib** | Visualization of the final Decision Tree structure |

## 🗂️ Repository Contents

| File/Folder | Description |
| :--- | :--- |
| `SCT_DS_3.py` | The core Python script containing data preprocessing, model training, hyperparameter tuning, and evaluation logic. |
| `bank-full.csv` | The raw dataset used for the project (45,000+ customer records). |
| `README.md` | This file. |
| `optimized_decision_tree.png` | The generated visualization of the final Decision Tree structure. |

## ⚙️ Data Preprocessing & Feature Engineering

The following steps were crucial for preparing the data for the classifier:

* **Target Encoding:** The target variable (`y`) was converted from 'yes'/'no' to **1/0** (binary classification).
* **Categorical Encoding:** All categorical features (e.g., `job`, `marital`, `education`) were converted to numerical format using **One-Hot Encoding** (`pd.get_dummies`).
* **Hyperparameter Tuning:** **Grid Search** was employed to optimize parameters like `max_depth` and `min_samples_leaf` using `roc_auc` scoring, ensuring the model generalizes well and avoids overfitting.

## 📈 Key Findings & Model Insights

The optimized Decision Tree provided clear, actionable rules for predicting subscription success:

* **Primary Predictor (Feature Importance):** The **duration of the last contact** (`duration`) was identified as the single most critical factor influencing a purchase decision.
* **Strong Success Rule:** Customers who had a **long last contact duration** (e.g., > 265 seconds) AND a **successful outcome from a previous campaign** (`poutcome_success`=1) have the highest probability of subscribing.
* **Recommendation:** Marketing efforts should prioritize customers based on these two key behavioral indicators.

## 🚀 How to Run the Project

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/vijaykaragulikv/SCT_DS_3.git](https://github.com/vijaykaragulikv/SCT_DS_3.git)
    cd SCT_DS_3
    ```

2.  **Install Dependencies:**
    ```bash
    pip install pandas scikit-learn matplotlib
    ```

3.  **Execute the Script:**
    ```bash
    python SCT_DS_3.py
    ```
The script will print the training progress, best parameters, and the final classification report to the console. It will also display and save the Decision Tree visualization.
