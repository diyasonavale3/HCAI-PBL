import os
import pandas as pd

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
