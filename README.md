# 📩 **SMS Spam Detection**

This repository contains the implementation of an **SMS Spam Classifier** model that detects whether a given message is **spam** or **ham** (not spam).  
It uses **machine learning algorithms** along with extensive **preprocessing and feature engineering techniques** to achieve **high accuracy**.

# 📚 **Project Overview**

The objective of this project is to build a **robust and accurate SMS spam classifier** using various machine learning models and compare their performances.

# 🔗 **The Workflow Includes:**

- **Data Loading and Exploration**
- **Data Cleaning and Preprocessing**
- **Feature Engineering**
- **Text Preprocessing (Tokenization, Stopword Removal, Stemming)**
- **Feature Vectorization (Bag of Words)**
- **Model Training and Evaluation**
- **Hyperparameter Tuning**
- **Final Model Selection and Saving**
- **ROC Curve Analysis**

# 🛠️ **Tech Stack and Libraries**

- **Python**
- **Pandas, NumPy** — Data manipulation
- **Matplotlib, Seaborn** — Data visualization
- **NLTK** — Natural Language Processing
- **Scikit-learn** — Machine Learning Models and Metrics
- **XGBoost** — Gradient Boosting Model
- **Joblib** — Model Serialization

# 📝 **Dataset**

- **Source:** SMS Spam Collection Dataset
- **Details:** 5,572 SMS messages labeled as "**ham**" (legitimate) or "**spam**".

# 🔥 **Project Workflow**

## 1. **Data Loading and Exploration**
- Dataset loaded using **Pandas**.
- **Exploratory Data Analysis (EDA)** to understand the distribution and characteristics of data.

## 2. **Data Preprocessing**
- Dropped unnecessary columns.
- Renamed columns for clarity.
- **Label encoding** applied (**ham → 0, spam → 1**).
- Removed duplicates and checked for missing values.

## 3. **Feature Engineering**
- **num_characters:** Number of characters in a message.
- **num_words:** Number of words in a message.
- **num_sentences:** Estimated number of sentences.

## 4. **Text Preprocessing**
- Lowercasing
- Tokenization
- Removal of punctuation and stopwords
- Stemming using **PorterStemmer**

## 5. **Feature Vectorization**
- **Bag of Words (BoW)** model using **CountVectorizer** (top 3000 frequent words).

## 6. **Model Training and Evaluation**

**Models Trained:**
- **Logistic Regression**
- **Support Vector Machine (LinearSVC)**
- **Multinomial Naive Bayes, Bernoulli Naive Bayes**
- **Decision Tree**
- **Random Forest**
- **AdaBoost**
- **Gradient Boosting**
- **XGBoost**

**Evaluation Metrics:**
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC AUC Score**

## 7. **Hyperparameter Tuning**
- Used **RandomizedSearchCV** for Logistic Regression, SVM, Decision Tree, Random Forest, AdaBoost, and XGBoost.

## 8. **Final Model Selection**
- **Logistic Regression** and **XGBoost** were selected based on best performance.
- Saved models and **CountVectorizer** using **joblib**.

## 9. **ROC Curve Analysis**
- **ROC Curves plotted.**
- **Logistic Regression AUC:** 0.99
- **XGBoost AUC:** 0.98

# 📈 **Results**

- **Best Models:** Logistic Regression and XGBoost
- **Final Model:** Logistic Regression (slightly better AUC)
- **Deployment Ready:** Models and vectorizer saved for real-world usage.

# 📈 **Results**

- **Best Models:** Logistic Regression and XGBoost
- **Final Model:** Logistic Regression (slightly better AUC)
- **Deployment Ready:** Models and vectorizer saved for real-world usage.

# 🏃‍♂️ **How to Run the Project**

Follow these steps to run the project locally:

### 1. **Clone the repository**

```bash
git clone https://github.com/your-username/sms-spam-detection.git
cd sms-spam-detection
### 1. **Clone the repository**
```
### 2. **Install the Required Dependencies**
```bash
pip install -r requirements.txt
```
### 3. **Run The Project**
```bash
python app.py
