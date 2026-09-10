"""
Task 5 setup: precompute everything the live interactive page needs, so
requests stay fast (no re-downloading data, re-vectorizing, or retraining
the base classifier on every click).

"""
import json
import os
import joblib
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from expert import simulate_expert, CLASS_NAMES

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
POOL_SIZE = 300


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
    train_df, test_df = load_data()

    # --- same proxy-classifier / signal-pool split as Task 3/4 ---
    proxy_train_df, signal_df = train_test_split(
        train_df, test_size=0.2, random_state=0, stratify=train_df['label'])

    proxy_vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2),
                                        stop_words='english', sublinear_tf=True)
    X_proxy_train = proxy_vectorizer.fit_transform(proxy_train_df['text'])
    proxy_model = LogisticRegression(C=3, max_iter=1000)
    proxy_model.fit(X_proxy_train, proxy_train_df['label'])

    X_signal = proxy_vectorizer.transform(signal_df['text'])
    signal_proba = proxy_model.predict_proba(X_signal)
    classifier_uncertainty = 1.0 - signal_proba.max(axis=1)

    # pick the POOL_SIZE most classifier-uncertain articles as the human-facing pool
    signal_df = signal_df.reset_index(drop=True)
    order = np.argsort(-classifier_uncertainty)[:POOL_SIZE]

    vectorizer = joblib.load(os.path.join(ARTIFACT_DIR, 'task1_vectorizer.joblib'))
    classifier = joblib.load(os.path.join(ARTIFACT_DIR, 'task1_model.joblib'))

    pool_texts = signal_df.loc[order, 'text'].tolist()
    pool_true = signal_df.loc[order, 'label'].to_numpy()
    X_pool = vectorizer.transform(pool_texts)
    pool_classifier_pred = classifier.predict(X_pool)
    pool_classifier_proba = classifier.predict_proba(X_pool).max(axis=1)

    pool = []
    for i, (text, true_label, pred_label, conf) in enumerate(
            zip(pool_texts, pool_true, pool_classifier_pred, pool_classifier_proba)):
        pool.append({
            'id': i,
            'text': text,
            'true_label': CLASS_NAMES[true_label],
            'classifier_pred': CLASS_NAMES[pred_label],
            'classifier_confidence': round(float(conf), 4),
        })
    with open(os.path.join(ARTIFACT_DIR, 'task5_pool.json'), 'w') as f:
        json.dump(pool, f, indent=2)

    # --- test set cache, so the live page never re-touches the raw dataset ---
    y_test = test_df['label'].to_numpy()
    X_test = vectorizer.transform(test_df['text'])
    classifier_pred_test = classifier.predict(X_test)
    expert_pred_test = simulate_expert(y_test, seed=0)  # same as Task 2/3/4

    cache = {
        'X_pool': X_pool,
        'X_test': X_test,
        'y_test': y_test,
        'classifier_pred_test': classifier_pred_test,
        'expert_pred_test': expert_pred_test,
        'classifier_accuracy': accuracy_score(y_test, classifier_pred_test),
    }
    joblib.dump(cache, os.path.join(ARTIFACT_DIR, 'task5_cache.joblib'))

    print(f'Pool saved: {len(pool)} articles')
    print(f'Cache saved: X_pool={X_pool.shape}, X_test={X_test.shape}')
    print(f'Classifier accuracy (reference): {cache["classifier_accuracy"]:.4f}')


if __name__ == '__main__':
    main()
