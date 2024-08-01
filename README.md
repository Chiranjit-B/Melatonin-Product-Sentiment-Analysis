# Melatonin Products Sentiment Analysis Project 🎉🧪

![Project Badge](https://img.shields.io/badge/Project-Melatonin%20Sentiment%20Analysis-brightgreen)



## Overview 📝

This project aims to analyze the market for melatonin products, which includes various doses such as 1mg, 2mg, 5mg, 10mg, etc. The main objectives are to:

- **Analyze the proportion of various products based on doses.**
- **Perform sentiment analysis on product reviews.**
- **Build predictive models to classify the dose of a product based on review features.**

## Project Structure 📂

- **Data Loading and Preparation**: Loading multiple CSV files containing product data.
- **Data Cleaning and Merging**: Cleaning the data and merging it into a single DataFrame.
- **Feature Extraction**: Extracting useful features from the reviews, such as sentiment scores and TF-IDF features.
- **Model Training and Evaluation**: Training machine learning models (Logistic Regression, Random Forest, and Gradient Boosting) and evaluating their performance.

## Setup and Installation 🚀

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/username/melatonin-sentiment-analysis.git
    cd melatonin-sentiment-analysis
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Jupyter Notebook**:
    ```bash
    jupyter notebook Chiranjit_Banerjee_Melatonin_Products_Sentiment_Analysis_Project.ipynb
    ```

## Data Loading 📥

The data consists of several CSV files for different melatonin products, which are loaded into Pandas DataFrames.

```python
v1 = "/path/to/file1.csv"
v1_Puritans = pd.read_csv(v1)
```

## Data Cleaning and Merging 🧹

Cleaning the data involves removing unnecessary columns, handling missing values, and merging the DataFrames into a single DataFrame.

```python
# Merge all dataframes into one
df_merged = pd.concat([v1_Puritans, v2_Carlyle300, v3_Carlyle180, v4_Natrol, v5_VitamaticMelatonin, v6_vitafusionMaxStrength, v7_ZzzQuilPure], ignore_index=True)
```

## Feature Extraction 🔍

### Sentiment Analysis

Using TextBlob to perform sentiment analysis on the review text. TextBlob is a Python library for processing textual data. It provides a simple API for diving into common natural language processing (NLP) tasks, such as part-of-speech tagging, noun phrase extraction, sentiment analysis, and more.

```python
df_merged['Sentmnt_Val'] = df_merged['review_text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
```

### Hot Words

Hot words are the most frequently occurring significant words in the reviews, used for feature extraction. Examples include 'melatonin', 'sleep', 'effect'.

### Stop Words

Stop words are common words that are usually filtered out because they carry less meaningful information. Examples include 'and', 'the', 'is'.

### TF-IDF Vectorization

Using TfidfVectorizer to convert the processed reviews into numerical features. TF-IDF (Term Frequency-Inverse Document Frequency) is a statistical measure used to evaluate the importance of a word in a document relative to a collection of documents (corpus). It helps in transforming text data into a format that machine learning algorithms can work with.

```python
vectorizer = TfidfVectorizer(max_features=100, max_df=0.8, stop_words='english')
A_train_tfidf = vectorizer.fit_transform(A_train['new_reviewed_text'])
```

## Model Training and Evaluation 🏋️‍♂️

### Logistic Regression

Performing hyperparameter tuning and training a Logistic Regression model. Logistic Regression is a statistical method for predicting binary outcomes from data. It is widely used in machine learning for binary classification tasks.

```python
param_grid_logreg = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                     'penalty': ['l2'],
                     'solver': ['lbfgs', 'newton-cg', 'sag'],
                     'max_iter': [100, 500, 1000, 1500]}

grid_search_logreg = GridSearchCV(LogisticRegression(), param_grid_logreg, cv=5)
grid_search_logreg.fit(A_train_features_scaled, B_train)
best_logreg_model = grid_search_logreg.best_estimator_

# Predict with the best Logistic Regression model
B_pred_logreg = best_logreg_model.predict(A_test_features_scaled)

# Evaluate Logistic Regression model
accuracy_logreg = accuracy_score(B_test, B_pred_logreg)
print("Accuracy :", accuracy_logreg)
print("Classification Report :\n", classification_report(B_test, B_pred_logreg))
```

### Random Forest

Training a Random Forest model and evaluating its performance. Random Forest is an ensemble learning method that constructs multiple decision trees and merges their outputs to improve accuracy and control over-fitting.

```python
rf_model = RandomForestClassifier()
rf_model.fit(A_train_features, B_train)

# Predict and evaluate
B_pred_rf = rf_model.predict(A_test_features)
accuracy_rf = accuracy_score(B_test, B_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)
print("Classification Report :\n", classification_report(B_test, B_pred_rf))
```

### Gradient Boosting

Training a Gradient Boosting model and evaluating its performance. Gradient Boosting is an ensemble technique that builds models sequentially, with each new model correcting errors made by the previous ones. It is effective for both classification and regression tasks.

```python
from sklearn.ensemble import GradientBoostingClassifier

param_grid_gb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

grid_search_gb = GridSearchCV(GradientBoostingClassifier(), param_grid_gb, cv=5)
grid_search_gb.fit(A_train_features_scaled, B_train)
best_gb_model = grid_search_gb.best_estimator_

# Predict with the best Gradient Boosting model
B_pred_gb = best_gb_model.predict(A_test_features_scaled)

# Evaluate Gradient Boosting model
accuracy_gb = accuracy_score(B_test, B_pred_gb)
print("Gradient Boosting Accuracy:", accuracy_gb)
print("Classification Report :\n", classification_report(B_test, B_pred_gb))
```

## Evaluation Metrics 📊

- **Accuracy**: Measures overall correctness of the model.
- **Precision**: Measures the accuracy of positive predictions.
- **Recall**: Measures the ability to identify all actual positives.
- **F1 Score**: Balances precision and recall.
- **Support**: Number of actual occurrences of each class.

## Results 🏆

| Metric           | Logistic Regression | Random Forest | Gradient Boosting |
|------------------|---------------------|---------------|-------------------|
| **Accuracy**     | 0.85                | 0.88          | 0.89              |
| **Precision**    | 0.84                | 0.87          | 0.88              |
| **Recall**       | 0.85                | 0.88          | 0.89              |
| **F1 Score**     | 0.84                | 0.88          | 0.88              |

## Conclusions 📈

- The Gradient Boosting model outperforms both the Random Forest and Logistic Regression models in terms of accuracy.
- TF-IDF vectorization and sentiment analysis significantly contribute to the model's performance.
- The project demonstrates the effective use of natural language processing and machine learning for product review analysis.

## Techniques and Algorithms Used 🧠

- **Natural Language Processing (NLP)**: Used for text preprocessing, tokenization, stopword removal, and sentiment analysis.
- **TF-IDF Vectorization**: Converts text data into numerical features based on term frequency and inverse document frequency.
- **Machine Learning Algorithms**: Logistic Regression, Random Forest, and Gradient Boosting for classification tasks.
- **Hyperparameter Tuning**: GridSearchCV for finding the best model parameters.
- **Data Standardization**: StandardScaler for normalizing feature values.

## Acknowledgments 🙏

- **Libraries Used**:
    - Pandas
    - NumPy
    - TextBlob
    - NLTK
    - scikit-learn
    - Matplotlib
    - Seaborn
![image](https://github.com/user-attachments/assets/0a6358e0-047b-4a34-adc8-7082067f7cfd)

