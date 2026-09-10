"""
Task 3: Learning to defer.

Builds a "rejector" model that decides, from article text alone, whether
to trust the Task 1 classifier or defer to the Task 2 simulated expert.

To avoid teaching the rejector an overly optimistic view of the classifier
(it scores ~4pts higher on its own training data than on new data), we
train a throwaway "proxy" classifier on 80% of the training set and use
its honest, out-of-sample predictions on the remaining 20% to build the
rejector's training labels. The real Task 1 classifier (trained on all
120k articles) is still what actually gets deployed/evaluated.

Run with: python project3/experiments/task3_defer.py
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

    # --- Step 1: proxy classifier, trained on 80%, evaluated on the other 20% ---
    proxy_train_df, signal_df = train_test_split(
        train_df, test_size=0.2, random_state=0, stratify=train_df['label'])

    proxy_vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2),
                                        stop_words='english', sublinear_tf=True)
    X_proxy_train = proxy_vectorizer.fit_transform(proxy_train_df['text'])
    proxy_model = LogisticRegression(C=3, max_iter=1000)
    proxy_model.fit(X_proxy_train, proxy_train_df['label'])

    X_signal = proxy_vectorizer.transform(signal_df['text'])
    signal_classifier_pred = proxy_model.predict(X_signal)

    # --- Step 2: build "should have deferred" labels on the held-out 20% ---
    signal_y = signal_df['label'].to_numpy()
    signal_expert_pred = simulate_expert(signal_y, seed=1)

    classifier_correct = signal_classifier_pred == signal_y
    expert_correct = signal_expert_pred == signal_y
    defer_target = (expert_correct & ~classifier_correct).astype(int)
    print(f'Rejector training signal: {len(signal_df)} examples, '
          f'{defer_target.mean():.1%} labeled "should defer"')

    # --- Step 3: train the rejector (reuses the REAL Task 1 vectorizer) ---
    vectorizer = joblib.load(os.path.join(ARTIFACT_DIR, 'task1_vectorizer.joblib'))
    classifier = joblib.load(os.path.join(ARTIFACT_DIR, 'task1_model.joblib'))

    X_signal_real_vec = vectorizer.transform(signal_df['text'])
    rejector = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced')
    rejector.fit(X_signal_real_vec, defer_target)

    # --- Step 4: deploy on the test set ---
    y_test = test_df['label'].to_numpy()
    X_test = vectorizer.transform(test_df['text'])

    classifier_pred_test = classifier.predict(X_test)
    expert_pred_test = simulate_expert(y_test, seed=0)  
    defer_decision_test = rejector.predict(X_test)

    final_pred = np.where(defer_decision_test == 1, expert_pred_test, classifier_pred_test)

    team_accuracy = accuracy_score(y_test, final_pred)
    classifier_accuracy = accuracy_score(y_test, classifier_pred_test)
    expert_accuracy = accuracy_score(y_test, expert_pred_test)
    deferral_rate = float(defer_decision_test.mean())

    # oracle: best possible outcome if we deferred exactly when it truly helped
    oracle_correct = (classifier_pred_test == y_test) | (expert_pred_test == y_test)
    oracle_accuracy = float(oracle_correct.mean())

    # random-deferral baseline at the same rate, averaged over 20 draws
    rng = np.random.default_rng(0)
    random_accs = []
    n_defer = int(round(deferral_rate * len(y_test)))
    for _ in range(20):
        idx = rng.choice(len(y_test), size=n_defer, replace=False)
        random_final = classifier_pred_test.copy()
        random_final[idx] = expert_pred_test[idx]
        random_accs.append(accuracy_score(y_test, random_final))
    random_baseline_accuracy = float(np.mean(random_accs))

    # quality of deferral decisions: among deferred cases, how often did it help?
    deferred_mask = defer_decision_test == 1
    n_deferred = int(deferred_mask.sum())
    if n_deferred > 0:
        deferred_classifier_correct = float((classifier_pred_test[deferred_mask] == y_test[deferred_mask]).mean())
        deferred_expert_correct = float((expert_pred_test[deferred_mask] == y_test[deferred_mask]).mean())
    else:
        deferred_classifier_correct = deferred_expert_correct = None

    print(f'\nClassifier alone:        {classifier_accuracy:.4f}')
    print(f'Expert alone:            {expert_accuracy:.4f}')
    print(f'Team (learned deferral): {team_accuracy:.4f}')
    print(f'Random deferral (same rate): {random_baseline_accuracy:.4f}')
    print(f'Oracle ceiling:          {oracle_accuracy:.4f}')
    print(f'Deferral rate: {deferral_rate:.1%}  ({n_deferred} of {len(y_test)} articles)')
    if n_deferred > 0:
        print(f'  Among deferred: classifier would have been right {deferred_classifier_correct:.1%} of the time')
        print(f'  Among deferred: expert was right {deferred_expert_correct:.1%} of the time')

    joblib.dump(rejector, os.path.join(ARTIFACT_DIR, 'task3_rejector.joblib'))
    metrics = {
        'team_accuracy': team_accuracy,
        'classifier_accuracy': classifier_accuracy,
        'expert_accuracy': expert_accuracy,
        'random_baseline_accuracy': random_baseline_accuracy,
        'oracle_accuracy': oracle_accuracy,
        'deferral_rate': deferral_rate,
        'n_deferred': n_deferred,
        'n_test': len(y_test),
        'deferred_classifier_correct_rate': deferred_classifier_correct,
        'deferred_expert_correct_rate': deferred_expert_correct,
        'rejector_train_defer_fraction': float(defer_target.mean()),
    }
    with open(os.path.join(ARTIFACT_DIR, 'task3_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    main()
