import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from django.conf import settings
from django.shortcuts import render, redirect
from .forms import DatasetUploadForm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from .forms import TrainingForm

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

    target_is_numeric = pd.api.types.is_numeric_dtype(df[target])
    task_note = None
    if task == 'regression' and not target_is_numeric:
        task = 'classification'
        task_note = "Regression needs a numeric target, but this dataset's target is categorical, so it's shown as classification instead."

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
        'task_note': task_note,
        'target_is_numeric': target_is_numeric,
        'detected': detect_task(df),
        'image_url': settings.MEDIA_URL + filename + '?t=' + str(os.path.getmtime(image_path)),
    }
    return render(request, 'project1/visualize.html', context)


def make_model(name, setting, task):
    if task == 'classification':
        if name == 'linear':
            return LogisticRegression(C=setting, max_iter=2000)
        if name == 'tree':
            return DecisionTreeClassifier(max_depth=setting, random_state=0)
        return RandomForestClassifier(max_depth=setting, random_state=0)
    if name == 'linear':
        return Ridge(alpha=setting)
    if name == 'tree':
        return DecisionTreeRegressor(max_depth=setting, random_state=0)
    return RandomForestRegressor(max_depth=setting, random_state=0)

def score_model(model, X, y, task):
    if task == 'classification':
        return accuracy_score(y, model.predict(X))
    return r2_score(y, model.predict(X))

def train(request):
    path = request.session.get('dataset_path')
    if not path:
        return redirect('project1:upload')

    df = load_dataset(path)
    features = list(df.columns[:-1])
    target = df.columns[-1]
    task = detect_task(df)

    form = TrainingForm(request.GET or None)
    context = {'form': form, 'target': target, 'task': task,
               'metric': 'accuracy' if task == 'classification' else 'R squared'}

    if request.GET and form.is_valid():
        name = form.cleaned_data['model']
        test_fraction = form.cleaned_data['test_size'] / 100
        X = df[features]
        y = df[target]
        stratify = y if task == 'classification' else None

        X_rest, X_test, y_rest, y_test = train_test_split(
            X, y, test_size=test_fraction, random_state=0, stratify=stratify)
        stratify_rest = y_rest if task == 'classification' else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_rest, y_rest, test_size=0.25, random_state=0, stratify=stratify_rest)

        if name == 'linear':
            values = [0.01, 0.1, 1, 10, 100]
            param_label = 'C' if task == 'classification' else 'alpha'
        else:
            values = [1, 2, 3, 5, 10]
            param_label = 'max_depth'

        results = []
        for setting in values:
            model = make_model(name, setting, task)
            model.fit(X_train, y_train)
            results.append({
                'setting': setting,
                'train': round(score_model(model, X_train, y_train, task), 3),
                'val': round(score_model(model, X_val, y_val, task), 3),
                'model': model,
            })

        best = max(results, key=lambda r: r['val'])
        final_score = round(score_model(best['model'], X_test, y_test, task), 3)

        context.update({
            'results': [{k: r[k] for k in ('setting', 'train', 'val')} for r in results],
            'best_setting': best['setting'],
            'best_val': best['val'],
            'final_score': final_score,
            'param_label': param_label,
            'n_train': len(X_train),
            'n_val': len(X_val),
            'n_test': len(X_test),
        })

    return render(request, 'project1/train.html', context)
