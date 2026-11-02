# Sentiment Analysis of Social Media Posts - Machine Learning Project

Course project, Machine Learning, B.Sc. Industrial Engineering & Management, Ben-Gurion University.

**Goal:** classify ~40K social media posts as positive or negative, using the post text together with post and user metadata (likes, followers, account age, checkmark type, posting time, hashtags).

## Pipeline

1. **EDA** - class balance, missing values, distributions by post type, timezone, posting hour, account type and age group; top hashtags and TF-IDF terms per sentiment.
2. **Pre-processing** - type conversion, duplicate checks, missing-value imputation, age and time-of-day categories, segmentation features (embedded content, profile picture).
3. **Feature engineering** - one-hot encoding, Min-Max scaling, hashtag counts, TF-IDF based positive/negative word scores, n-grams reduced to 300 components with Truncated SVD.
4. **Feature selection** - Fisher score and mutual information.
5. **Models** (5-fold stratified cross-validation, AUC-ROC):

| Model | Validation AUC |
|---|---|
| Logistic Regression (baseline) | 0.944 |
| Decision Tree (grid search) | 0.91 |
| Linear SVM (grid search on C) | 0.939 |
| MLP (grid search on layers, learning rate, batch size, activation) | 0.952 |
| MLP + BERT sentence embeddings (all-MiniLM-L6-v2) | 0.971 |
| **Final: Keras network + BERT embeddings, dropout, early stopping** | **0.978** |

6. **Clustering** - K-Medoids with Manhattan, Gower and Euclidean distances, evaluated with Silhouette, Davies-Bouldin, ARI and NMI against the sentiment labels.
7. **Test set** - the final model predicts sentiment for an unlabeled test set.

## Notebooks

- `1_eda_preprocessing_features.ipynb` - EDA, pre-processing, feature extraction and selection
- `2_models_clustering_improvements.ipynb` - model training and tuning, clustering, BERT and dropout improvements
- `3_test_predictions.ipynb` - applying the final model to the test set

The notebooks were run in Google Colab. The course dataset is not included.

**Tools:** Python, pandas, scikit-learn, TensorFlow/Keras, Sentence-Transformers (BERT), scikit-learn-extra, matplotlib, seaborn.
