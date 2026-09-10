# HCAI-PBL

Coursework for Human-Centric Artificial Intelligence — a single Django project
hosting a separate app per assignment.

## Group
- Diya Raju Sonavale (674186)

## Projects
- [Project 1](project1/README.md) Automated Machine Learning: upload a CSV, visualize it and train a supervised learning model.
- [Project 2](project2/README.md) Explainability: decision tree / logistic regression with an interpretability complexity trade off, counterfactual explanations and PDP/ALE feature effect plots on the Palmer Penguins dataset.
- [Project 3](project3/report.md) Active Learning for Learning-to-Defer: a classifier that can defer to a simulated human expert on AG News topic classification, plus an active-learning strategy for querying that expert under a limited budget. Full report: [project3/static/project3/report.pdf](project3/static/project3/report.pdf).
- [Project 4](project4/report.md) Preference Elicitation: a Bradley-Terry/ Plackett-Luce preference model over the IMDB 5000 Movie Dataset, with two interactive elicitation interfaces (pairwise comparison and 10 movie ranking) and a designed user study comparing them. Full report: [project4/static/project4/report.pdf](project4/static/project4/report.pdf).

## Setup
1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the server:
   ```bash
   python manage.py runserver
   ```
4. Open http://127.0.0.1:8000/home/ in your browser. It lists all four projects with links into each one.

## Tech Stack
Django, pandas, numpy, scikit-learn, matplotlib, palmerpenguins, datasets (Hugging Face)
