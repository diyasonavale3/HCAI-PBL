import os
import json
import joblib
import numpy as np
from django.conf import settings
from django.shortcuts import render, redirect
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

ARTIFACT_DIR = os.path.join(settings.BASE_DIR, 'project3', 'artifacts')
CLASS_NAMES = ['World', 'Sports', 'Business', 'Sci/Tech']

POOL = None
CACHE = None


def load_pool():
    global POOL
    if POOL is None:
        with open(os.path.join(ARTIFACT_DIR, 'task5_pool.json')) as f:
            POOL = json.load(f)
    return POOL


def load_cache():
    global CACHE
    if CACHE is None:
        CACHE = joblib.load(os.path.join(ARTIFACT_DIR, 'task5_cache.joblib'))
    return CACHE


def index(request):
    return render(request, 'project3/index.html')


def build_rejector(pool, queried_ids, human_labels, X_pool):
    """Train the rejector from human-provided labels so far. Returns None
    until we have at least one "should defer" and one "should not defer"
    example to learn from."""
    if not queried_ids:
        return None
    id_to_pos = {p['id']: i for i, p in enumerate(pool)}
    rows = np.array([id_to_pos[i] for i in queried_ids])
    classifier_correct = np.array([pool[id_to_pos[i]]['classifier_pred'] == pool[id_to_pos[i]]['true_label']
                                    for i in queried_ids])
    human_correct = np.array([lbl == pool[id_to_pos[i]]['true_label']
                               for i, lbl in zip(queried_ids, human_labels)])
    defer_target = (human_correct & ~classifier_correct).astype(int)
    if len(set(defer_target.tolist())) < 2:
        return None
    rejector = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced')
    rejector.fit(X_pool[rows], defer_target)
    return rejector


def pick_next_id(pool, queried_ids, X_pool, rejector):
    remaining = [p['id'] for p in pool if p['id'] not in queried_ids]
    if not remaining:
        return None
    if rejector is None:
        remaining.sort(key=lambda i: pool[i]['classifier_confidence'])
        return remaining[0]
    id_to_pos = {p['id']: i for i, p in enumerate(pool)}
    remaining_rows = np.array([id_to_pos[i] for i in remaining])
    proba = rejector.predict_proba(X_pool[remaining_rows])[:, 1]
    pick_pos = int(np.argmin(np.abs(proba - 0.5)))
    return remaining[pick_pos]


def evaluate_team(cache, rejector):
    y_test = cache['y_test']
    classifier_pred_test = cache['classifier_pred_test']
    expert_pred_test = cache['expert_pred_test']
    if rejector is None:
        defer_decision = np.zeros(len(y_test), dtype=int)
    else:
        defer_decision = rejector.predict(cache['X_test'])
    final_pred = np.where(defer_decision == 1, expert_pred_test, classifier_pred_test)
    team_accuracy = float(accuracy_score(y_test, final_pred))
    deferral_rate = float(defer_decision.mean())
    return team_accuracy, deferral_rate


def task5(request):
    pool = load_pool()
    cache = load_cache()
    X_pool = cache['X_pool']

    queried = request.session.get('t5_queried', [])
    labels = request.session.get('t5_labels', [])
    history = request.session.get('t5_history', [])

    if request.method == 'POST':
        pending_id = request.session.get('t5_pending')
        answer = request.POST.get('label')
        if pending_id is not None and answer in CLASS_NAMES:
            queried.append(pending_id)
            labels.append(answer)
            request.session['t5_queried'] = queried
            request.session['t5_labels'] = labels

            rejector = build_rejector(pool, queried, labels, X_pool)
            team_accuracy, deferral_rate = evaluate_team(cache, rejector)
            history.append({
                'n_queried': len(queried),
                'team_accuracy': round(team_accuracy, 4),
                'deferral_rate': round(deferral_rate, 4),
            })
            request.session['t5_history'] = history

            id_to_pos = {p['id']: i for i, p in enumerate(pool)}
            article = pool[id_to_pos[pending_id]]
            request.session['t5_feedback'] = {
                'text': article['text'],
                'true_label': article['true_label'],
                'classifier_pred': article['classifier_pred'],
                'human_label': answer,
                'human_correct': answer == article['true_label'],
                'classifier_correct': article['classifier_pred'] == article['true_label'],
            }
            request.session['t5_pending'] = None
        return redirect('project3:task5')

    # GET
    feedback = request.session.pop('t5_feedback', None)
    pending_id = request.session.get('t5_pending')

    if pending_id is None:
        rejector = build_rejector(pool, queried, labels, X_pool)
        pending_id = pick_next_id(pool, queried, X_pool, rejector)
        request.session['t5_pending'] = pending_id

    id_to_pos = {p['id']: i for i, p in enumerate(pool)}
    pending_article = pool[id_to_pos[pending_id]] if pending_id is not None else None

    context = {
        'feedback': feedback,
        'pending_article': pending_article,
        'n_queried': len(queried),
        'pool_size': len(pool),
        'history': history[-10:],
        'latest_team_accuracy': history[-1]['team_accuracy'] if history else None,
        'classifier_accuracy': round(cache['classifier_accuracy'], 4),
        'class_names': CLASS_NAMES,
    }
    return render(request, 'project3/task5.html', context)


def task5_reset(request):
    for key in ['t5_queried', 't5_labels', 't5_pending', 't5_feedback', 't5_history']:
        request.session.pop(key, None)
    return redirect('project3:task5')
