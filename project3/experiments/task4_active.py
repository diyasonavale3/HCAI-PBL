"""
Task 4: Active learning for expert competence discovery.

Reuses the exact same 24k "signal" pool that Task 3 held out from the
proxy classifier (so if we spent the whole budget, we'd reproduce Task 3's
result exactly a useful sanity check). Instead of using the expert's
answer on all 24k, we query a limited budget, chosen either:
  - "active": round 0 picks the classifier's least-confident articles;
    every later round picks whatever the *current rejector* is most
    unsure whether to defer on (directly targets learning the deferral
    boundary, not just general classification difficulty)
  - "random": same budget schedule, but queries are picked uniformly at
    random each round the baseline we compare against

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
BATCH_SIZE = 500
N_ROUNDS = 8
TARGET_DEFER_RATE = 0.06254166666666666  # exact population rate, from Task 3



def clean_text(text):
    return text.replace('\\', ' ')


def load_data():
    ds = load_dataset('fancyzhx/ag_news')
    train_df = ds['train'].to_pandas()
    test_df = ds['test'].to_pandas()
    train_df['text'] = train_df['text'].map(clean_text)
    test_df['text'] = test_df['text'].map(clean_text)
    return train_df, test_df


def run_strategy(strategy, calibrate, signal_y, signal_classifier_pred, signal_expert_pred,
                  classifier_uncertainty, X_signal_real_vec, X_test, y_test,
                  classifier_pred_test, expert_pred_test, seed):
    n_pool = len(signal_y)
    rng = np.random.default_rng(seed)
    queried = np.zeros(n_pool, dtype=bool)
    results = []
    rejector = None

    for round_idx in range(N_ROUNDS):
        remaining = np.where(~queried)[0]
        if strategy == 'random':
            batch = rng.choice(remaining, size=BATCH_SIZE, replace=False)
        elif round_idx == 0:
            order = remaining[np.argsort(-classifier_uncertainty[remaining])]
            batch = order[:BATCH_SIZE]
        else:
            proba = rejector.predict_proba(X_signal_real_vec[remaining])[:, 1]
            rejector_uncertainty = -np.abs(proba - 0.5)
            order = remaining[np.argsort(-rejector_uncertainty)]
            batch = order[:BATCH_SIZE]

        queried[batch] = True
        q_idx = np.where(queried)[0]

        classifier_correct = signal_classifier_pred[q_idx] == signal_y[q_idx]
        expert_correct = signal_expert_pred[q_idx] == signal_y[q_idx]
        defer_target = (expert_correct & ~classifier_correct).astype(int)

        rejector = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced')
        rejector.fit(X_signal_real_vec[q_idx], defer_target)

        test_proba = rejector.predict_proba(X_test)[:, 1]
        if calibrate:
            train_proba = rejector.predict_proba(X_signal_real_vec[q_idx])[:, 1]
            threshold = np.quantile(train_proba, 1 - TARGET_DEFER_RATE)
            defer_decision_test = (test_proba >= threshold).astype(int)
        else:
            defer_decision_test = rejector.predict(X_test)

        final_pred = np.where(defer_decision_test == 1, expert_pred_test, classifier_pred_test)
        team_accuracy = accuracy_score(y_test, final_pred)

        results.append({
            'n_queried': int(len(q_idx)),
            'team_accuracy': float(team_accuracy),
            'deferral_rate': float(defer_decision_test.mean()),
            'defer_target_positive_rate': float(defer_target.mean()),
        })
        label = 'calibrated' if calibrate else 'uncalibrated'
        print(f'[{strategy}/{label}] round {round_idx+1}/{N_ROUNDS}  '
              f'n_queried={len(q_idx):5d}  team_acc={team_accuracy:.4f}')

    return results


def main():
    train_df, test_df = load_data()

    # --- exact same proxy-classifier / signal-pool split as Task 3 ---
    proxy_train_df, signal_df = train_test_split(
        train_df, test_size=0.2, random_state=0, stratify=train_df['label'])

    proxy_vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2),
                                        stop_words='english', sublinear_tf=True)
    X_proxy_train = proxy_vectorizer.fit_transform(proxy_train_df['text'])
    proxy_model = LogisticRegression(C=3, max_iter=1000)
    proxy_model.fit(X_proxy_train, proxy_train_df['label'])

    X_signal = proxy_vectorizer.transform(signal_df['text'])
    signal_classifier_pred = proxy_model.predict(X_signal)
    signal_proba = proxy_model.predict_proba(X_signal)
    classifier_uncertainty = 1.0 - signal_proba.max(axis=1)

    signal_y = signal_df['label'].to_numpy()
    signal_expert_pred = simulate_expert(signal_y, seed=1)  # same as Task 3

    # --- real deployed classifier + vectorizer (Task 1) ---
    vectorizer = joblib.load(os.path.join(ARTIFACT_DIR, 'task1_vectorizer.joblib'))
    classifier = joblib.load(os.path.join(ARTIFACT_DIR, 'task1_model.joblib'))
    X_signal_real_vec = vectorizer.transform(signal_df['text'])

    y_test = test_df['label'].to_numpy()
    X_test = vectorizer.transform(test_df['text'])
    classifier_pred_test = classifier.predict(X_test)
    expert_pred_test = simulate_expert(y_test, seed=0)  # same as Task 2/3

    classifier_accuracy = accuracy_score(y_test, classifier_pred_test)
    expert_accuracy = accuracy_score(y_test, expert_pred_test)

    print('=== Active strategy (uncalibrated) ===')
    active_uncal = run_strategy('active', False, signal_y, signal_classifier_pred, signal_expert_pred,
                                 classifier_uncertainty, X_signal_real_vec, X_test, y_test,
                                 classifier_pred_test, expert_pred_test, seed=0)

    print('\n=== Random strategy (uncalibrated) ===')
    random_uncal = run_strategy('random', False, signal_y, signal_classifier_pred, signal_expert_pred,
                                 classifier_uncertainty, X_signal_real_vec, X_test, y_test,
                                 classifier_pred_test, expert_pred_test, seed=2)

    print('\n=== Active strategy (calibrated) ===')
    active_cal = run_strategy('active', True, signal_y, signal_classifier_pred, signal_expert_pred,
                               classifier_uncertainty, X_signal_real_vec, X_test, y_test,
                               classifier_pred_test, expert_pred_test, seed=0)

    print('\n=== Random strategy (calibrated) ===')
    random_cal = run_strategy('random', True, signal_y, signal_classifier_pred, signal_expert_pred,
                               classifier_uncertainty, X_signal_real_vec, X_test, y_test,
                               classifier_pred_test, expert_pred_test, seed=2)

    metrics = {
        'classifier_accuracy': classifier_accuracy,
        'expert_accuracy': expert_accuracy,
        'task3_full_pool_team_accuracy': 0.9322,
        'batch_size': BATCH_SIZE,
        'n_rounds': N_ROUNDS,
        'target_defer_rate': TARGET_DEFER_RATE,
        'active_results_uncalibrated': active_uncal,
        'random_results_uncalibrated': random_uncal,
        'active_results_calibrated': active_cal,
        'random_results_calibrated': random_cal,
    }

    with open(os.path.join(ARTIFACT_DIR, 'task4_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    main()
