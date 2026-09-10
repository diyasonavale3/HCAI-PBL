import os
import json
import random
from django.conf import settings
from django.shortcuts import render, redirect

from .models import StudySession, PairwiseTrial, RankingTrial

ARTIFACT_DIR = os.path.join(settings.BASE_DIR, 'project4', 'artifacts')
N_PAIRWISE_TRIALS = 20
N_RANKING_ROUNDS = 5
RANKING_SET_SIZE = 10

MOVIES = None


def load_movies():
    global MOVIES
    if MOVIES is None:
        with open(os.path.join(ARTIFACT_DIR, 'movies.json')) as f:
            data = json.load(f)
        MOVIES = data['movies']
    return MOVIES


def get_session(request):
    sid = request.session.get('study_session_id')
    if sid is None:
        return None
    try:
        return StudySession.objects.get(pk=sid)
    except StudySession.DoesNotExist:
        return None


def next_step_url(session_obj):
    if session_obj.order == 'pairwise_first':
        if not session_obj.pairwise_complete:
            return 'project4:pairwise'
        elif not session_obj.ranking_complete:
            return 'project4:ranking'
    else:
        if not session_obj.ranking_complete:
            return 'project4:ranking'
        elif not session_obj.pairwise_complete:
            return 'project4:pairwise'
    return 'project4:complete'


def index(request):
    return render(request, 'project4/index.html')


def start_study(request):
    order = random.choice(['pairwise_first', 'ranking_first'])
    session_obj = StudySession.objects.create(order=order)
    request.session['study_session_id'] = session_obj.pk
    return redirect(next_step_url(session_obj))


def pairwise(request):
    session_obj = get_session(request)
    if session_obj is None:
        return redirect('project4:index')
    movies = load_movies()

    if request.method == 'POST':
        PairwiseTrial.objects.create(
            session=session_obj,
            trial_number=int(request.POST['trial_number']),
            movie_a_id=int(request.POST['movie_a_id']),
            movie_b_id=int(request.POST['movie_b_id']),
            chosen_movie_id=int(request.POST['chosen']),
        )
        if session_obj.pairwise_trials.count() >= N_PAIRWISE_TRIALS:
            session_obj.pairwise_complete = True
            session_obj.save()
        return redirect(next_step_url(session_obj))

    count = session_obj.pairwise_trials.count()
    if count >= N_PAIRWISE_TRIALS:
        return redirect(next_step_url(session_obj))

    movie_a, movie_b = random.sample(movies, 2)
    context = {
        'movie_a': movie_a, 'movie_b': movie_b,
        'trial_number': count + 1, 'total_trials': N_PAIRWISE_TRIALS,
    }
    return render(request, 'project4/pairwise.html', context)


def ranking(request):
    session_obj = get_session(request)
    if session_obj is None:
        return redirect('project4:index')
    movies = load_movies()

    if request.method == 'POST':
        RankingTrial.objects.create(
            session=session_obj,
            round_number=int(request.POST['round_number']),
            movie_ids=[int(x) for x in request.POST['movie_ids'].split(',')],
            ranking=[int(x) for x in request.POST['ranking'].split(',')],
        )
        if session_obj.ranking_trials.count() >= N_RANKING_ROUNDS:
            session_obj.ranking_complete = True
            session_obj.save()
        return redirect(next_step_url(session_obj))

    count = session_obj.ranking_trials.count()
    if count >= N_RANKING_ROUNDS:
        return redirect(next_step_url(session_obj))

    sample = random.sample(movies, RANKING_SET_SIZE)
    context = {
        'movies': sample, 'round_number': count + 1, 'total_rounds': N_RANKING_ROUNDS,
        'movie_ids_csv': ','.join(str(m['id']) for m in sample),
    }
    return render(request, 'project4/ranking.html', context)


def complete(request):
    return render(request, 'project4/complete.html')
