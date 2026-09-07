# Project 2: Explainability

## Group
- Diya Raju Sonavale

## Overview
A Django app exploring model interpretability on the Palmer Penguins dataset (species classification from bill/flipper/body measurements, island, sex and year).
Covers global structure (decision tree vs logistic regression, with an accuracy vs complexity trade off), local explanations (counterfactuals) and global feature effects (PDP and ALE plots),
all linked to the same model type and λ selection.

## Features
- **Task 1**: Fit a decision tree, showing the tree diagram, test accuracy and number of leaves.
- **Task 2**: A λ slider trades off accuracy against tree complexity, trains a family of trees at different `max_leaf_nodes`, then picks and displays whichever one maximises `max_leaf_nodes`.
- **Task 3**: The same λ trade off for logistic regression using the L1 norm of the (standardised) coefficients as the complete measure Ω(f),
  shown as a pre species coefficient bar chart. The λ sliders range adapts automatically to whichever model type is selected since leaf counts and coefficient norms live on very different numeric scales.
- **Task 4**: A "Counterfactual" region- Pick an example penguin and a target species and see up to 5 small realistic perturbations of that penguin
  (Gaussian noise on numeric features and random resampling on island/year/sex) that the currently selected model would classify as the target ranked by MAD weighted L1 distance.
  Sampling escalates (more points, more noise) if nothing is found on the first attempt.
- **Task 5**: A "Feature Effect Plots" region- pick one of the four numeric features and see both a aPDP and an ALE plot (one curve per species) for its effect on the currently selected models predicted probabilities.
  ALE is computed via finite differences for the decision tree (whose predictions are a non-differentiable step function) and via the exact analytic partial derivative of the softmax output for logistic regression
  (verified against numerical differentiation to ~le-13).

## Tech Stack
  Django, pandas, numpy, scikit-learn, matplotlib, palmerpenguins
  
## Setup
  See the [root README](../README.md) for environment setup. Then open
  http://127.0.0.1:8000/project2/ in your browser.

## Usage
1. Choose a model type (Decision tree or logistic regression) and a λ value.
2. Review the resulting models accuracy, complexity and diagram/coefficients.
3. In the Counterfactual section, pick an example penguin and a target species to see nearby examples the model would classify differently.
4. In the feature effect plots section, pick a numeric feature to see its PDP and ALE curves.

