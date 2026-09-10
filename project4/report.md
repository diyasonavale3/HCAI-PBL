# Project 4: Preference Elicitation

## Group
- Diya Raju Sonavale (674186)

## Introduction

This project designs a preference elicitation system for movie recommendation, built on a Bradley-Terry-style latent utility model `U(x) = wᵀx`. Two interaction designs for
eliciting a user's preference vector `w` are implemented and compared: pairwise comparison, and ranking sets of ten movies. This report covers the feature representation (Task 1), the mathematical extension of Bradley-Terry needed to handle full rankings (Task 2) and the design of a user study to compare the two elicitation methods (Task 3). The corresponding interactive interface (Task 4) is available
directly from the project's landing page.

## Task 1: Feature Representation

**Dataset.** The IMDB 5000 Movie Dataset (5,043 movies, 28 metadata fields). After removing 126 duplicate titles and 118 rows missing core fields (year, duration, score, or genre), 4,799 movies remained.

**Feature design.** Since `w` must be estimated from only a handful of user interactions, the feature space needs to be small and semantically meaningful rather than high dimensional and sparse. Specific actor/director identities were deliberately excluded (thousands of near unique values, unlearnable from a few comparisons); instead, 22 features were extracted per movie:

- **12 genre indicators** (multi hot): Drama, Comedy, Thriller, Action, Romance, Adventure, Crime, Sci-Fi, Fantasy, Horror, Family, Mystery, the 12 most frequent genres in the dataset (each occurring 500+ times; the next most common, Biography, drops to 293).
- **5 content-rating indicators** (one-hot): G, PG, PG-13, R, and "Other" (covering the long tail of rare/missing ratings, Not Rated, TV-14, NC-17, etc. which together cover only 13% of movies and would otherwise create many near empty categories).
- **5 standardized numeric features**: release year, duration, IMDb score, log-scaled vote count (a popularity proxy), and log-scaled total cast Facebook likes (a coarse "star power" proxy that avoids needing per-actor identity weights).

All numeric features are standardized (zero mean, unit variance) across the dataset so that no single feature dominates the utility function purely due to its raw numeric
scale.

## Task 2: Extending Bradley-Terry to Rankings

**Standard Bradley-Terry** defines the probability that movie *i* is preferred to movie *j* as a function of their utility difference:

`P(i ≻ j) = exp(U(i)) / (exp(U(i)) + exp(U(j)))`

This handles a single pairwise comparison, but Design 2 collects full rankings of 10 items, `i₁ ≻ i₂ ≻ ... ≻ iₙ`, which the standard formulation does not model.

**Proposed extension: the Plackett-Luce model.** A ranking is modeled as a sequence of independent "choose the best of what remains" decisions:

`P(i₁ ≻ i₂ ≻ ... ≻ iₙ) = ∏ₖ₌₁ⁿ [ exp(U(iₖ)) / Σₗ₌ₖⁿ exp(U(iₗ)) ]`

Concretely: the top ranked item is chosen from all *n* candidates with probability proportional to its exponentiated utility relative to all *n* (a softmax); *given* that
choice, the second ranked item is chosen from the remaining *n−1* the same way; and so on, recursively, until one item remains.

**Justification.** This is the natural generalization of Bradley-Terry, not an arbitrary alternative: setting *n = 2* collapses the formula exactly back to the original pairwise expression, so every property of the pairwise model is preserved as a special case. It is also the standard model for this exact situation in the preference learning literature. Practically, it means a single 10 item ranking can be decomposed into 9 sequential pairwise equivalent likelihood terms for the purpose of fitting `w` via maximum likelihood a property that directly motivated the efficiency
hypothesis in the Task 3 user study design.

## Task 3: User Study Design

### Hypothesis

The Task 2 math gives a natural, well motivated hypothesis: a single 10 movie ranking
decomposes (via Plackett-Luce) into 9 pairwise-equivalent comparisons, so *in principle* each ranking screen carries far more information than a single pairwise screen.

**H1 (primary)**: For the same amount of participant effort (a fixed number of screens), the ranking interface (Design 2) yields a more accurate estimate of a participant's latent preference vector `w` than the pairwise interface (Design 1).

**Secondary, exploratory questions**: Does one interface take less time per unit of information gained? Do participants subjectively prefer one? Is one perceived as more
mentally effortful?

### Design

- **Within-subjects**, as already built: every participant completes both Design 1 (20 pairwise trials) and Design 2 (5 rounds of ranking 10 movies), with presentation order randomized per participant to control for practice/fatigue effects (already implemented via the interface's random 50/50 order assignment).
- **One addition needed beyond what Task 4's interface currently does**: to actually *test* H1, we need a way to measure "how accurate is the resulting `w`?" which requires evaluating each method's fitted preference model against choices it never saw. So the complete protocol adds a short **held-out validation phase** *after* both main tasks: 10 additional pairwise comparisons, drawn from a fixed, curated set of movie pairs (not random but deliberately chosen to span the feature space, e.g. varying genre/era/rating combinations, so the evaluation set is informative rather than by chance easy or all-alike). This phase isn't part of Task 4's interface (which only needed to implement the two elicitation methods themselves), but a "complete" study needs it to actually adjudicate the hypothesis.
- **Independent variable**: elicitation method (pairwise vs. ranking), manipulated within-subject.
- **Dependent variables**: (1) held-out prediction accuracy [primary], (2) time taken per method, (3) subjective preference/perceived difficulty [secondary].

### Recruitment

- **Population**: general adult participants via an online panel (e.g. Prolific), screened for fluent English and being a regular moviegoer (self-reported: watches movies at least monthly) since preference judgments about unfamiliar films are noisy for someone with no relevant taste to draw on.
- **Sample size**: a paired comparison (Wilcoxon signed-rank or paired t-test) targeting a medium effect size (Cohen's d ≈ 0.5), α = 0.05, power = 0.8, needs **≈34 participants** the standard textbook figure for this design; a small pilot (n≈5–10) would be run first to sanity-check the actual effect size and timing estimates before committing to full recruitment.
- **Compensation**: a fair per-minute rate for the estimated ~15–20 minute session (20 pairwise + 5 rankings + 10 held-out comparisons + a brief questionnaire).
- **Exclusions**: failed attention checks, or completion times so fast they suggest random clicking rather than genuine judgments.

### Procedure (what would actually happen, step by step)

1. Recruit via the panel; screen for eligibility (English fluency, movie-watching frequency).
2. Informed consent: explain the task, data usage, anonymity, and voluntary withdrawal rights.
3. Random order assignment (pairwise-first or ranking-first)  already built into the interface.
4. Participant completes their first assigned method.
5. Participant completes their second assigned method.
6. Held-out validation phase: 10 fixed pairwise comparisons (same set for every participant, for comparability).
7. Brief post-study questionnaire: which method they preferred and why (open text), perceived difficulty of each (1–5 Likert scale), basic demographics (age range movie-watching frequency) for potential moderator analysis.
8. Debrief and compensation.

### Analysis plan

- For each participant, fit `ŵ_pairwise` and `ŵ_ranking` separately via maximum likelihood on that method's collected responses (Bradley-Terry likelihood for pairwise data, the Plackett-Luce likelihood from Task 2 for ranking data).
- Evaluate both fitted vectors on the *same* held-out 10 comparisons, scoring % correctly predicted.
- Compare the two accuracy distributions with a **Wilcoxon signed-rank test** (paired, robust to non-normal accuracy scores) the primary test of H1.
- Secondary: paired test on completion time; descriptive/binomial analysis of subjective preference; a check that order (which method came first) didn't itself drive the effect, to confirm counterbalancing worked as intended.

### Ethics

Standard informed consent, anonymized response storage (no data linking a participant's identity to their movie preferences beyond what's needed for compensation), institutional ethics/IRB approval prior to running and clearly communicated right to withdraw at any point without penalty.
