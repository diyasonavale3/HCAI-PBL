import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from django.conf import settings
from django.shortcuts import render
from palmerpenguins import load_penguins
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = ['island', 'bill_length_mm', 'bill_depth_mm',
                    'flipper_length_mm', 'body_mass_g', 'sex', 'year']
TARGET_COLUMN = 'species'
SPECIES = ['Adelie', 'Chinstrap', 'Gentoo']

CONTINUOUS_FEATURES = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
CATEGORICAL_FEATURES = ['island', 'sex', 'year']
ONE_HOT_COLUMNS = ['island', 'sex']

TREE_MAX_LEAF_NODES_RANGE = range(2, 21)
LOGISTIC_C_RANGE = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]

LAMBDA_BOUNDS = {
    'tree': {'min': 0.0, 'max': 0.2, 'step': 0.005},
    'logistic': {'min': 0.0, 'max': 0.08, 'step': 0.002},
}

CF_ATTEMPT_SCHEDULE = [
    {'n': 300, 'sigma_scale': 0.5, 'flip_prob': 0.2},
    {'n': 1000, 'sigma_scale': 1.0, 'flip_prob': 0.35},
    {'n': 3000, 'sigma_scale': 2.0, 'flip_prob': 0.5},
]
N_COUNTERFACTUALS = 5


def load_data():
    """Palmer Penguins, missing rows dropped, categoricals one-hot encoded."""
    df = load_penguins().dropna().reset_index(drop=True)
    X = pd.get_dummies(df[FEATURE_COLUMNS], columns=ONE_HOT_COLUMNS)
    y = df[TARGET_COLUMN]
    return df, X, y


def split_data(df, X, y):
    idx_train, idx_test = train_test_split(
        df.index, test_size=0.25, random_state=0, stratify=y)
    return (df.loc[idx_train], df.loc[idx_test].reset_index(drop=True),
             X.loc[idx_train], X.loc[idx_test],
             y.loc[idx_train], y.loc[idx_test])


def fit_tree_family(X_train, y_train, X_test, y_test):
    family = []
    for max_leaves in TREE_MAX_LEAF_NODES_RANGE:
        model = DecisionTreeClassifier(max_leaf_nodes=max_leaves, random_state=0)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        family.append({'model': model, 'scaler': None,
                        'omega': model.get_n_leaves(), 'accuracy': acc})
    return family


def fit_logistic_family(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    family = []
    for C in LOGISTIC_C_RANGE:
        model = LogisticRegression(C=C, max_iter=5000, random_state=0)
        model.fit(X_train_s, y_train)
        acc = accuracy_score(y_test, model.predict(X_test_s))
        omega = float(np.abs(model.coef_).sum())
        family.append({'model': model, 'scaler': scaler, 'omega': omega, 'accuracy': acc})
    return family


def select_best(family, lam):
    """Pick the model maximizing accuracy - lam * omega, tie-broken toward smaller omega."""
    return max(family, key=lambda m: (m['accuracy'] - lam * m['omega'], -m['omega']))


def selected_predict(selected, X_encoded):
    """Predict with whichever model won selection, applying its scaler if it has one."""
    X_input = selected['scaler'].transform(X_encoded) if selected['scaler'] is not None else X_encoded
    return selected['model'].predict(X_input)


def render_tree_plot(model, feature_names):
    filename = 'project2_tree.png'
    image_path = os.path.join(settings.MEDIA_ROOT, filename)
    fig, ax = plt.subplots(figsize=(18, 10))
    plot_tree(model, feature_names=feature_names, class_names=model.classes_,
              filled=True, fontsize=8, ax=ax)
    fig.savefig(image_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    return filename


def render_coef_plot(model, feature_names):
    filename = 'project2_coefs.png'
    image_path = os.path.join(settings.MEDIA_ROOT, filename)

    n_classes, n_features = model.coef_.shape
    x = np.arange(n_features)
    width = 0.8 / n_classes

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, class_name in enumerate(model.classes_):
        ax.bar(x + i * width, model.coef_[i], width, label=str(class_name))
    ax.set_xticks(x + width * (n_classes - 1) / 2)
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('Coefficient (standardized features)')
    ax.legend(title='species')
    fig.tight_layout()
    fig.savefig(image_path, dpi=100)
    plt.close(fig)
    return filename


def encode_like(raw_df, ref_columns):
    """One-hot encode raw feature rows and align columns to match a fitted model's input."""
    encoded = pd.get_dummies(raw_df[FEATURE_COLUMNS], columns=ONE_HOT_COLUMNS)
    return encoded.reindex(columns=ref_columns, fill_value=0)


def compute_feature_stats(raw_train):
    std = {f: raw_train[f].std() for f in CONTINUOUS_FEATURES}
    mad = {f: (raw_train[f] - raw_train[f].median()).abs().median() for f in CONTINUOUS_FEATURES}
    categories = {f: sorted(raw_train[f].dropna().unique().tolist()) for f in CATEGORICAL_FEATURES}
    return {'std': std, 'mad': mad, 'categories': categories}


def generate_perturbations(x_raw, n, sigma_scale, flip_prob, feature_stats, rng):
    """n noisy copies of x_raw: Gaussian jitter on continuous features, random
    resampling on low-cardinality ones."""
    rows = []
    for _ in range(n):
        new_row = x_raw.copy()
        for feat in CONTINUOUS_FEATURES:
            noise = rng.normal(0, sigma_scale * feature_stats['std'][feat])
            new_row[feat] = x_raw[feat] + noise
        for feat in CATEGORICAL_FEATURES:
            if rng.random() < flip_prob:
                new_row[feat] = rng.choice(feature_stats['categories'][feat])
        rows.append(new_row)
    return pd.DataFrame(rows).reset_index(drop=True)


def mad_weighted_distance(x_raw, candidate_raw, feature_stats):
    dist = 0.0
    for feat in CONTINUOUS_FEATURES:
        mad = feature_stats['mad'][feat] or 1e-6
        dist += abs(candidate_raw[feat] - x_raw[feat]) / mad
    for feat in CATEGORICAL_FEATURES:
        dist += 0.0 if candidate_raw[feat] == x_raw[feat] else 1.0
    return dist


def build_display_row(raw_row, distance=None, reference=None):
    """One table row: a cell per feature, flagging which differ from the reference."""
    cells = []
    for feat in FEATURE_COLUMNS:
        value = raw_row[feat]
        changed = reference is not None and raw_row[feat] != reference[feat]
        cells.append({'value': value, 'changed': changed})
    return {'cells': cells, 'distance': distance}


def find_counterfactuals(x_raw, target_class, selected, ref_columns, feature_stats, seed):
    rng = np.random.default_rng(seed)
    found = []
    attempts_used = 0
    for attempt in CF_ATTEMPT_SCHEDULE:
        attempts_used += 1
        candidates_raw = generate_perturbations(
            x_raw, attempt['n'], attempt['sigma_scale'], attempt['flip_prob'], feature_stats, rng)
        candidates_X = encode_like(candidates_raw, ref_columns)
        preds = selected_predict(selected, candidates_X)
        hits_raw = candidates_raw[preds == target_class]
        for _, row in hits_raw.iterrows():
            found.append((mad_weighted_distance(x_raw, row, feature_stats), row))
        if len(found) >= N_COUNTERFACTUALS:
            break
    found.sort(key=lambda item: item[0])
    return found[:N_COUNTERFACTUALS], attempts_used


NUM_EFFECT_BINS = 20


def build_grid(raw_reference, feature, n_bins=NUM_EFFECT_BINS):
    """Quantile-based bin edges, so every bin holds a similar number of real points."""
    edges = np.quantile(raw_reference[feature], np.linspace(0, 1, n_bins + 1))
    return np.unique(edges)


def get_predict_proba(selected):
    """Wrap the selected model's predict_proba, applying its scaler if it has one."""
    def predict_proba(X_encoded):
        X_input = selected['scaler'].transform(X_encoded) if selected['scaler'] is not None else X_encoded
        return selected['model'].predict_proba(X_input)
    return predict_proba


def compute_pdp(feature, grid, raw_reference, ref_columns, selected):
    """PDP(v, c) = average predicted P(class=c) after forcing every row's feature to v."""
    predict_proba = get_predict_proba(selected)
    classes = list(selected['model'].classes_)
    pdp = {c: [] for c in classes}
    for v in grid:
        modified = raw_reference.copy()
        modified[feature] = v
        probs = predict_proba(encode_like(modified, ref_columns))
        for ci, c in enumerate(classes):
            pdp[c].append(float(probs[:, ci].mean()))
    return pdp, classes


def finish_ale(local_effects, counts, classes):
    """Accumulate per-bin local effects into a curve, then center it to weighted-mean zero."""
    ale = {}
    total = counts.sum()
    for c in classes:
        accumulated = np.concatenate([[0.0], np.cumsum(local_effects[c])])
        midpoints = (accumulated[:-1] + accumulated[1:]) / 2
        weighted_mean = (midpoints * counts).sum() / total if total > 0 else 0.0
        ale[c] = (accumulated - weighted_mean).tolist()
    return ale


def compute_ale_tree(feature, grid, raw_reference, ref_columns, selected):
    """Finite-difference ALE: the tree's predictions are a step function, so there's
    no derivative to compute exactly - we test the two edges of each bin instead."""
    predict_proba = get_predict_proba(selected)
    classes = list(selected['model'].classes_)
    n_bins = len(grid) - 1
    values = raw_reference[feature].to_numpy()
    bin_idx = np.clip(np.digitize(values, grid[1:-1]), 0, n_bins - 1)

    local_effects = {c: np.zeros(n_bins) for c in classes}
    counts = np.zeros(n_bins)
    for k in range(n_bins):
        mask = bin_idx == k
        counts[k] = mask.sum()
        if counts[k] == 0:
            continue
        subset = raw_reference[mask]
        lower = subset.copy(); lower[feature] = grid[k]
        upper = subset.copy(); upper[feature] = grid[k + 1]
        p_lower = predict_proba(encode_like(lower, ref_columns))
        p_upper = predict_proba(encode_like(upper, ref_columns))
        for ci, c in enumerate(classes):
            local_effects[c][k] = (p_upper[:, ci] - p_lower[:, ci]).mean()

    return finish_ale(local_effects, counts, classes)


def compute_ale_logistic(feature, grid, raw_reference, ref_columns, selected):
    """Exact-derivative ALE: logistic regression's probability output is smooth, so we
    use the closed-form partial derivative of softmax instead of a finite difference."""
    model = selected['model']
    scaler = selected['scaler']
    classes = list(model.classes_)
    n_bins = len(grid) - 1
    j = ref_columns.index(feature)
    std_j = scaler.scale_[j]

    values = raw_reference[feature].to_numpy()
    bin_idx = np.clip(np.digitize(values, grid[1:-1]), 0, n_bins - 1)

    X_scaled = scaler.transform(encode_like(raw_reference, ref_columns))
    probs = model.predict_proba(X_scaled)
    coef = model.coef_

    # dP_c/dx_scaled_j = P_c * (w_c,j - sum_k P_k * w_k,j); convert to unstandardized units
    weighted_sum = probs @ coef[:, j]
    derivs = probs * (coef[:, j][None, :] - weighted_sum[:, None]) / std_j

    widths = np.diff(grid)
    local_effects = {c: np.zeros(n_bins) for c in classes}
    counts = np.zeros(n_bins)
    for k in range(n_bins):
        mask = bin_idx == k
        counts[k] = mask.sum()
        if counts[k] == 0:
            continue
        for ci, c in enumerate(classes):
            local_effects[c][k] = derivs[mask, ci].mean() * widths[k]

    return finish_ale(local_effects, counts, classes)


def render_effect_plot(feature, grid, pdp, ale, classes):
    filename = 'project2_effects.png'
    image_path = os.path.join(settings.MEDIA_ROOT, filename)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for c in classes:
        axes[0].plot(grid, pdp[c], marker='o', markersize=3, label=str(c))
    axes[0].set_title(f'PDP: {feature}')
    axes[0].set_xlabel(feature)
    axes[0].set_ylabel('Predicted probability')
    axes[0].legend(title='species')

    for c in classes:
        axes[1].plot(grid, ale[c], marker='o', markersize=3, label=str(c))
    axes[1].axhline(0, color='black', linewidth=0.8)
    axes[1].set_title(f'ALE: {feature}')
    axes[1].set_xlabel(feature)
    axes[1].set_ylabel('Accumulated local effect (centered)')
    axes[1].legend(title='species')

    fig.tight_layout()
    fig.savefig(image_path, dpi=100)
    plt.close(fig)
    return filename


def index(request):
    model_type = request.GET.get('model', 'tree')
    if model_type not in ('tree', 'logistic'):
        model_type = 'tree'
    bounds = LAMBDA_BOUNDS[model_type]

    try:
        lam = float(request.GET.get('lam', bounds['min']))
    except ValueError:
        lam = bounds['min']
    lam = round(min(max(lam, bounds['min']), bounds['max']), 4)

    df, X, y = load_data()
    raw_train, raw_test, X_train, X_test, y_train, y_test = split_data(df, X, y)
    feature_names = list(X.columns)

    if model_type == 'tree':
        family = fit_tree_family(X_train, y_train, X_test, y_test)
        best = select_best(family, lam)
        filename = render_tree_plot(best['model'], feature_names)
        omega_label = 'Number of leaves'
    else:
        family = fit_logistic_family(X_train, y_train, X_test, y_test)
        best = select_best(family, lam)
        filename = render_coef_plot(best['model'], feature_names)
        omega_label = 'L1 norm of coefficients'

    image_path = os.path.join(settings.MEDIA_ROOT, filename)
        # --- Feature effect plots ---
    effect_feature = request.GET.get('effect_feature', CONTINUOUS_FEATURES[0])
    if effect_feature not in CONTINUOUS_FEATURES:
        effect_feature = CONTINUOUS_FEATURES[0]

    grid = build_grid(raw_train, effect_feature)
    pdp, classes = compute_pdp(effect_feature, grid, raw_train, feature_names, best)
    if model_type == 'tree':
        ale = compute_ale_tree(effect_feature, grid, raw_train, feature_names, best)
    else:
        ale = compute_ale_logistic(effect_feature, grid, raw_train, feature_names, best)
    effect_filename = render_effect_plot(effect_feature, grid, pdp, ale, classes)
    effect_image_path = os.path.join(settings.MEDIA_ROOT, effect_filename)


    # --- Counterfactuals ---
    try:
        cf_index = int(request.GET.get('cf_index', 0))
    except ValueError:
        cf_index = 0
    cf_index = min(max(cf_index, 0), len(raw_test) - 1)

    x_raw = raw_test.loc[cf_index]
    default_target = next((s for s in SPECIES if s != x_raw[TARGET_COLUMN]), SPECIES[0])
    cf_target = request.GET.get('cf_target', default_target)
    if cf_target not in SPECIES:
        cf_target = default_target

    feature_stats = compute_feature_stats(raw_train)
    seed = hash((cf_index, cf_target, model_type, lam)) % (2**32)
    counterfactuals, attempts_used = find_counterfactuals(
        x_raw, cf_target, best, feature_names, feature_stats, seed)

    table_rows = [build_display_row(x_raw)]
    for dist, row in counterfactuals:
        table_rows.append(build_display_row(row, distance=round(dist, 3), reference=x_raw))

    test_options = [
        {'index': i, 'label': f"#{i}: {r[TARGET_COLUMN]} — bill {r['bill_length_mm']}mm, "
                                f"flipper {r['flipper_length_mm']}mm, {r['island']}"}
        for i, r in raw_test.iterrows()
    ]

    context = {
        'model_type': model_type,
        'accuracy': round(best['accuracy'], 3),
        'omega': round(best['omega'], 3),
        'omega_label': omega_label,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'lam': lam,
        'lam_min': bounds['min'],
        'lam_max': bounds['max'],
        'lam_step': bounds['step'],
        'image_url': settings.MEDIA_URL + filename + '?t=' + str(os.path.getmtime(image_path)),
        'feature_columns': FEATURE_COLUMNS,
        'test_options': test_options,
        'cf_index': cf_index,
        'cf_target': cf_target,
        'species_options': SPECIES,
        'table_rows': table_rows,
        'cf_found_count': len(counterfactuals),
        'attempts_used': attempts_used,
        'effect_feature': effect_feature,
        'continuous_features': CONTINUOUS_FEATURES,
        'effect_image_url': settings.MEDIA_URL + effect_filename + '?t=' + str(os.path.getmtime(effect_image_path)),

    }
    return render(request, 'project2/index.html', context)
