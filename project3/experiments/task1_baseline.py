"""
Task 1: Baseline classifier for AG News.

Hyperparameters (C=3, ngram_range=(1,2)) were chosen via a validation sweep
(see git history / report) see project3/experiments/ for that sweep.
This script trains the final model on the FULL training set and evaluates
once on the official test set.

Run with: python project3/experiments/task1_baseline.py
"""
import json
import os
import joblib
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
CLASS_NAMES = ['World', 'Sports', 'Business', 'Sci/Tech']


def clean_text(text):
    return text.replace('\\', ' ')


def load_data():
    ds = load_dataset('fancyzhx/ag_news')
    train_df = ds['train'].to_pandas()
    test_df = ds['test'].to_pandas()
    train_df['text'] = train_df['text'].map(clean_text)
    test_df['text'] = test_df['text'].map(clean_text)
    return train_df, test_df


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    train_df, test_df = load_data()

    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2),
                                  stop_words='english', sublinear_tf=True)
    X_train = vectorizer.fit_transform(train_df['text'])
    X_test = vectorizer.transform(test_df['text'])
    y_train = train_df['label']
    y_test = test_df['label']

    model = LogisticRegression(C=3, max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f'Test accuracy: {accuracy:.4f}')
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    joblib.dump(vectorizer, os.path.join(ARTIFACT_DIR, 'task1_vectorizer.joblib'))
    joblib.dump(model, os.path.join(ARTIFACT_DIR, 'task1_model.joblib'))

    metrics = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'class_names': CLASS_NAMES,
        'n_train': len(train_df),
        'n_test': len(test_df),
        'hyperparameters': {'C': 3, 'ngram_range': [1, 2], 'max_features': 50000},
    }
    with open(os.path.join(ARTIFACT_DIR, 'task1_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    main()
