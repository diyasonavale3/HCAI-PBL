"""
Shared simulated-expert module, reused across Task 2/3/4 experiments.

The expert is a class-conditional noisy labeler: for each article, with
probability EXPERT_ACCURACY[true_class] it returns the true label,
otherwise it returns one of the other three classes chosen uniformly at
random. This is deliberately complementary to the Task 1 baseline
classifier strong on Business/Sci-Tech (where the classifier struggles),
weak on Sports/World (where the classifier excels) so that "should I
defer?" is a genuinely non-trivial per-article decision in later tasks.
"""
import numpy as np

CLASS_NAMES = ['World', 'Sports', 'Business', 'Sci/Tech']
# indices: 0=World, 1=Sports, 2=Business, 3=Sci/Tech
EXPERT_ACCURACY = {0: 0.55, 1: 0.55, 2: 0.90, 3: 0.90}


def simulate_expert(labels, seed=0):
    """labels: array-like of true integer class labels (0-3).
    Returns an array of simulated expert predictions, same length."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    predictions = np.empty_like(labels)
    for i, y in enumerate(labels):
        if rng.random() < EXPERT_ACCURACY[y]:
            predictions[i] = y
        else:
            others = [c for c in range(4) if c != y]
            predictions[i] = rng.choice(others)
    return predictions
