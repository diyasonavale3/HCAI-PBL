"""
Task 1: Feature representation for the IMDB 5000 Movie Dataset.

Downloads the dataset, cleans it, and extracts a compact, semantically
meaningful feature vector per movie: genre indicators, content-rating
category, and standardized numeric attributes (year, duration, rating,
popularity, star power). Deliberately excludes high-cardinality identity
features (specific actor/director names) since preference elicitation
must estimate w from only a handful of user interactions.

"""
import json
import os
import io
import urllib.request
import numpy as np
import pandas as pd

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
DATA_URL = 'https://raw.githubusercontent.com/sundeepblue/movie_rating_prediction/master/movie_metadata.csv'

TOP_GENRES = ['Drama', 'Comedy', 'Thriller', 'Action', 'Romance', 'Adventure',
              'Crime', 'Sci-Fi', 'Fantasy', 'Horror', 'Family', 'Mystery']
RATING_BUCKETS = ['G', 'PG', 'PG-13', 'R']  # anything else -> 'Other'


def load_raw():
    req = urllib.request.Request(DATA_URL, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read()
    return pd.read_csv(io.BytesIO(data))


def bucket_rating(value):
    if value in RATING_BUCKETS:
        return value
    return 'Other'


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    df = load_raw()

    df = df.drop_duplicates(subset='movie_title', keep='first')
    df = df.dropna(subset=['title_year', 'duration', 'imdb_score', 'genres'])
    df = df.reset_index(drop=True)

    df['content_rating'] = df['content_rating'].apply(bucket_rating)
    df['genre_list'] = df['genres'].str.split('|')
    df['num_voted_users'] = df['num_voted_users'].fillna(0)
    df['cast_total_facebook_likes'] = df['cast_total_facebook_likes'].fillna(0)

    # --- numeric features, standardized ---
    numeric_raw = pd.DataFrame({
        'title_year': df['title_year'],
        'duration': df['duration'],
        'imdb_score': df['imdb_score'],
        'log_num_voted_users': np.log1p(df['num_voted_users']),
        'log_cast_facebook_likes': np.log1p(df['cast_total_facebook_likes']),
    })
    numeric_mean = numeric_raw.mean()
    numeric_std = numeric_raw.std()
    numeric_scaled = (numeric_raw - numeric_mean) / numeric_std

    # --- genre multi-hot ---
    genre_features = pd.DataFrame({
        f'genre_{g}': df['genre_list'].apply(lambda gl: 1.0 if g in gl else 0.0)
        for g in TOP_GENRES
    })

    # --- content rating one-hot ---
    rating_categories = RATING_BUCKETS + ['Other']
    rating_features = pd.DataFrame({
        f'rating_{r}': (df['content_rating'] == r).astype(float)
        for r in rating_categories
    })

    feature_names = (list(genre_features.columns) + list(rating_features.columns)
                      + list(numeric_scaled.columns))
    feature_matrix = pd.concat([genre_features, rating_features, numeric_scaled], axis=1)[feature_names]

    movies = []
    for i in range(len(df)):
        movies.append({
            'id': i,
            'title': df.loc[i, 'movie_title'].strip(),
            'year': int(df.loc[i, 'title_year']),
            'genres': df.loc[i, 'genre_list'],
            'imdb_score': float(df.loc[i, 'imdb_score']),
            'duration': int(df.loc[i, 'duration']),
            'content_rating': df.loc[i, 'content_rating'],
            'features': feature_matrix.loc[i].tolist(),
        })

    with open(os.path.join(ARTIFACT_DIR, 'movies.json'), 'w') as f:
        json.dump({
            'feature_names': feature_names,
            'numeric_mean': numeric_mean.to_dict(),
            'numeric_std': numeric_std.to_dict(),
            'movies': movies,
        }, f, indent=2)

    print(f'Saved {len(movies)} movies with {len(feature_names)} features each.')
    print('Feature names:', feature_names)


if __name__ == '__main__':
    main()
