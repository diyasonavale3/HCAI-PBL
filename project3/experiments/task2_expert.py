"""
Task 2: Analyze the simulated expert's accuracy and strengths/weaknesses
on the AG News test set.

Run with: python project3/experiments/task2_expert.py
"""
import json
import os
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from expert import simulate_expert, CLASS_NAMES

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), '..', 'artifacts')


def clean_text(text):
    return text.replace('\\', ' ')


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    ds = load_dataset('fancyzhx/ag_news')
    test_df = ds['test'].to_pandas()
    test_df['text'] = test_df['text'].map(clean_text)

    y_test = test_df['label'].to_numpy()
    expert_pred = simulate_expert(y_test, seed=0)

    accuracy = accuracy_score(y_test, expert_pred)
    report = classification_report(y_test, expert_pred, target_names=CLASS_NAMES, output_dict=True)
    cm = confusion_matrix(y_test, expert_pred).tolist()

    print(f'Expert overall test accuracy: {accuracy:.4f}')
    print(classification_report(y_test, expert_pred, target_names=CLASS_NAMES))

    metrics = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'class_names': CLASS_NAMES,
        'n_test': len(test_df),
    }
    with open(os.path.join(ARTIFACT_DIR, 'task2_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    main()
