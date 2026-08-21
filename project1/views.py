import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from django.conf import settings
from django.shortcuts import render, redirect
from .forms import DatasetUploadForm

UPLOAD_DIR = os.path.join(settings.MEDIA_ROOT, 'uploads')


def load_dataset(path):
    """Read a CSV: first row = feature names, last column = target."""
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [str(c).strip() for c in df.columns]
    first = df.columns[0]
    if first.lower() in ('id', 'index', 'unnamed: 0'):
        df = df.drop(columns=[first])
    return df


def detect_task(df):
    target = df.columns[-1]
    if df[target].dtype == object or df[target].nunique() <= 20:
        return 'classification'
    return 'regression'


def index(request):
    return redirect('project1:upload')


def upload(request):
    error = None
    if request.method == 'POST':
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            path = os.path.join(UPLOAD_DIR, file.name)
            with open(path, 'wb') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)
            try:
                load_dataset(path)
            except Exception as e:
                error = f"Could not read this file as a CSV: {e}"
            else:
                request.session['dataset_path'] = path
                request.session['dataset_name'] = file.name
                return redirect('project1:preview')
    else:
        form = DatasetUploadForm()
    return render(request, 'project1/upload.html', {'form': form, 'error': error})


def preview(request):
    path = request.session.get('dataset_path')
    if not path or not os.path.exists(path):
        return redirect('project1:upload')

    df = load_dataset(path)
    target = df.columns[-1]
    features = list(df.columns[:-1])

    stats = df.describe().round(3)
    stats_rows = [[i] + list(r) for i, r in zip(stats.index, stats.values)]

    context = {
        'name': request.session.get('dataset_name'),
        'n_rows': len(df),
        'n_features': len(features),
        'features': features,
        'target': target,
        'task': detect_task(df),
        'n_classes': df[target].nunique(),
        'columns': list(df.columns),
        'rows': df.head(10).values.tolist(),
        'stats_columns': ['statistic'] + list(stats.columns),
        'stats_rows': stats_rows,
    }
    return render(request, 'project1/preview.html', context)


def visualize(request):
    path = request.session.get('dataset_path')
    if not path or not os.path.exists(path):
        return redirect('project1:upload')

    df = load_dataset(path)
    target = df.columns[-1]
    features = list(df.columns[:-1])

    task = request.GET.get('task') or detect_task(df)
    x = request.GET.get('x') or features[0]
    y = request.GET.get('y') or (features[1] if len(features) > 1 else target)

    if x not in df.columns or y not in df.columns:
        x, y = features[0], target

    filename = 'project1_scatter.png'
    image_path = os.path.join(settings.MEDIA_ROOT, filename)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    if task == 'classification':
        for value in sorted(df[target].unique()):
            subset = df[df[target] == value]
            ax.scatter(subset[x], subset[y], label=str(value), alpha=0.8)
        ax.legend(title=target)
    else:
        points = ax.scatter(df[x], df[y], c=df[target], cmap='viridis', alpha=0.8)
        fig.colorbar(points, ax=ax, label=target)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{y} vs {x}")
    fig.tight_layout()
    fig.savefig(image_path, dpi=110)
    plt.close(fig)

    context = {
        'features': features,
        'columns': list(df.columns),
        'target': target,
        'x': x,
        'y': y,
        'task': task,
        'detected': detect_task(df),
        'image_url': settings.MEDIA_URL + filename + '?t=' + str(os.path.getmtime(image_path)),
    }
    return render(request, 'project1/visualize.html', context)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from .forms import TrainingForm


def make_model(name, setting):
    if name == 'logistic':
        return LogisticRegression(C=setting, max_iter=2000)
    if name == 'tree':
        return DecisionTreeClassifier(max_depth=setting, random_state=0)
    return RandomForestClassifier(max_depth=setting, random_state=0)


def train(request):
    path = request.session.get('dataset_path')
    if not path:
        return redirect('project1:upload')

    df = load_dataset(path)
    features = list(df.columns[:-1])
    target = df.columns[-1]

    form = TrainingForm(request.GET or None)
    context = {'form': form, 'target': target}

    if request.GET and form.is_valid():
        name = form.cleaned_data['model']
        test_size = form.cleaned_data['test_size'] / 100

        X_train, X_test, y_train, y_test = train_test_split(
            df[features], df[target], test_size=test_size, random_state=0)

        settings_to_try = [0.01, 0.1, 1, 10, 100] if name == 'logistic' else [1, 2, 3, 5, 10]

        results = []
        for setting in settings_to_try:
            model = make_model(name, setting)
            model.fit(X_train, y_train)
            score = accuracy_score(y_test, model.predict(X_test))
            results.append({'setting': setting, 'score': round(score, 3)})

        context['results'] = results
        context['best'] = max(results, key=lambda r: r['score'])
        context['n_train'] = len(X_train)
        context['n_test'] = len(X_test)

    return render(request, 'project1/train.html', context)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from .forms import TrainingForm


def make_model(name, setting):
    if name == 'logistic':
        return LogisticRegression(C=setting, max_iter=2000)
    if name == 'tree':
        return DecisionTreeClassifier(max_depth=setting, random_state=0)
    return RandomForestClassifier(max_depth=setting, random_state=0)


def train(request):
    path = request.session.get('dataset_path')
    if not path:
        return redirect('project1:upload')

    df = load_dataset(path)
    features = list(df.columns[:-1])
    target = df.columns[-1]

    form = TrainingForm(request.GET or None)
    context = {'form': form, 'target': target}

    if request.GET and form.is_valid():
        name = form.cleaned_data['model']
        test_size = form.cleaned_data['test_size'] / 100

        X_train, X_test, y_train, y_test = train_test_split(
            df[features], df[target], test_size=test_size, random_state=0)

        settings_to_try = [0.01, 0.1, 1, 10, 100] if name == 'logistic' else [1, 2, 3, 5, 10]

        results = []
        for setting in settings_to_try:
            model = make_model(name, setting)
            model.fit(X_train, y_train)
            score = accuracy_score(y_test, model.predict(X_test))
            results.append({'setting': setting, 'score': round(score, 3)})

        context['results'] = results
        context['best'] = max(results, key=lambda r: r['score'])
        context['n_train'] = len(X_train)
        context['n_test'] = len(X_test)

    return render(request, 'project1/train.html', context)
