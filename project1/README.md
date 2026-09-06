# Project 1: Automated Machine Learning (Supervised Learning Interface)

## Group
- Diya Raju Sonavale (674186)

## Overview
A Django app for a basic supervised learning workflow: 
1) Upload a CSV dataset
2) visualise it
3) train a machine learning model on it

## Features
- **Task 1 & 2**: Home page listing group info and a link into this app.
- **Task 3**: Upload a CSV (first row = feature names, last column = target),
  preview the data (row/column counts, summary statistics),
  and visualise it as a scatter plot,
  coloured by class for classification or by a continuous target with a colour bar for regression.
- **Task 4**: Choose a model (linear/logistic regression, decision tree or random forest),
  split the data into train/validation/test sets, sweep across several hyper parameter values and see the best setting plus its score on a held out test set.

## Tech stack
Django, pandas, scikit-learn, matplotlib

## Setup
1) Create and activate a virtual environment:
    ```bash
   python -m venv venv
   source venv/bin/activate

2) Install dependencies:
   pip install django pandas scikit-learn matplotlib

3) Run the server:
   python manage.py runserver

4) Open http://127.0.0.1:8000/home/ in your browser.
