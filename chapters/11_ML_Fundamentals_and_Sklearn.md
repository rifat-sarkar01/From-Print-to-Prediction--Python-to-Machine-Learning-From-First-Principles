# Chapter 11: Machine Learning Fundamentals

## 11.1 What Machine Learning Actually Is

Traditional programming: you write explicit rules, the computer applies them to data, and you get an answer.
```
Rules + Data → Answers
```
Machine learning inverts this: you show the computer example data *and* the correct answers, and it works out the rules (a mathematical function) that map one to the other.
```
Data + Answers → Rules (the "model")
```
That learned function — the model — can then be applied to *new* data it has never seen, to predict answers for it.

## 11.2 The Three Main Categories

- **Supervised learning** — you have labeled data (inputs *and* correct outputs). The model learns to map input → output.
  - **Regression**: predicting a continuous number (house price, temperature).
  - **Classification**: predicting a category (spam/not spam, cat/dog/bird).
- **Unsupervised learning** — you only have inputs, no labels. The model finds structure on its own.
  - **Clustering**: grouping similar data points (customer segments).
  - **Dimensionality reduction**: compressing many features into fewer, while keeping the important structure.
- **Reinforcement learning** — an agent learns by taking actions in an environment and receiving rewards or penalties (used in game-playing AI, robotics). Not covered in depth here, but worth knowing the term.

## 11.3 The Real-World ML Workflow

Tutorials tend to skip straight to `model.fit()`. Real projects don't — most of the work happens before and after that one line:

```
Problem  →  Data  →  Cleaning  →  EDA  →  Feature Engineering
   →  Train / Validation / Test Split  →  Baseline
   →  Training  →  Evaluation  →  Hyperparameter Tuning
   →  Final Model  →  Deployment  →  Monitoring
```

- **Problem**: what decision will this model actually inform? A vague goal ("predict churn") produces a vague, hard-to-evaluate model — a good problem statement names the action taken on the prediction.
- **Data → Cleaning → EDA**: Chapters 8-9 — load it, fix it, look at it before touching a model.
- **Feature engineering**: building new, more informative columns from what you have (Chapter 8.11's `FamilySize` from `SibSp + Parch` is a small example).
- **Train / validation / test split**: covered in 11.4 below — three sets, not two.
- **Baseline**: the dumb model you must beat before anything fancier counts as progress (11.10).
- **Training → Evaluation → Tuning**: Chapter 12, in depth.
- **Deployment → Monitoring**: Chapter 14 — a model isn't "done" when it trains well; it's done when it's serving real predictions and someone is watching whether it keeps working.

Keep this diagram in view through Chapters 11-14 — each chapter fills in one or two boxes.

## 11.4 Splitting Data: Train, Validation, and Test

Two-way train/test splits are fine for a quick experiment, but the moment you start **tuning** a model (Chapter 12.13) — trying different settings and picking the best one — a two-way split quietly breaks. Here's why, and the fix.

- **Training set**: what the model learns from.
- **Validation set**: what you use to compare different models or settings *while you're still deciding*.
- **Test set**: touched exactly once, at the very end, to report how the final model will likely perform in the real world.

**Why three sets?** If you tune hyperparameters by repeatedly checking performance on your "test" set and picking whatever scores best, you've turned that test set into a second training set — you're now overfitting to it, just more slowly. The validation set absorbs that repeated checking; the test set stays untouched and honest.

```python
from sklearn.model_selection import train_test_split

# first split off the test set
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# then split what's left into train and validation
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42)
# result: 60% train, 20% validation, 20% test
```
In practice, many projects use **k-fold cross-validation** (Chapter 12.12) on the training data instead of a single fixed validation set — same purpose, more reliable, since it doesn't depend on which rows happened to land in the validation split.

## 11.5 Overfitting and Underfitting

- **Overfitting**: the model memorizes the training data too closely, including its noise and quirks, and performs poorly on new data. Think of a student who memorized the exact practice exam answers instead of learning the underlying concepts.
- **Underfitting**: the model is too simple to capture the real pattern, and performs poorly on *both* training and new data.

Here's what overfitting actually looks like when you print it — not an abstract warning, a real number gap:

```python
model = DecisionTreeClassifier(max_depth=None)   # unconstrained — free to memorize
model.fit(X_train, y_train)

print("Train accuracy:", model.score(X_train, y_train))   # 0.998
print("Test accuracy:", model.score(X_test, y_test))      # 0.712
```
**Common mistake:** looking only at training accuracy and shipping the model, because 99.8% *looks* great. A 28-point gap between train and test accuracy is a model that memorized rather than learned — it will disappoint in production almost immediately. The fix, in this case, is exactly what Chapter 12.3 already tells you: constrain the tree (`max_depth`, `min_samples_leaf`), or switch to a Random Forest, which resists this by design.

The train/test split (11.4) exists specifically to catch this gap. A model that scores 99% on training data but 71% on test data is clearly overfit, and you'd never know that if you only measured training performance.

## 11.6 Bias vs. Variance

This is the formal vocabulary behind "underfitting" and "overfitting" — worth having, since you'll see it in error messages, papers, and job interviews alike.

- **Bias** is error from a model that's too simple to capture the real pattern — systematically wrong in the same direction. High bias ≈ underfitting.
- **Variance** is error from a model that's too sensitive to the specific training data it happened to see — it would give a very different answer if trained on a slightly different sample. High variance ≈ overfitting.

**Intuition:** imagine training the same model type on 10 different random samples of your data. A high-bias model gives 10 similarly-wrong answers — consistent, but off-target. A high-variance model gives 10 wildly different answers — each one plausible on its own training set, but unstable.

This is the **bias-variance tradeoff**: reducing one tends to increase the other. A linear model has high bias but low variance (simple, stable, often wrong if the real pattern is curved). An unconstrained decision tree has low bias but high variance (flexible, unstable, prone to memorizing). Most of the "tuning knobs" you'll meet in Chapter 12 — `max_depth`, `n_estimators`, regularization strength — exist to let you dial along this tradeoff deliberately instead of leaving it to chance.

## 11.7 Scikit-learn's Unified Interface

Nearly every model in Scikit-learn (`sklearn`) follows the same three-method pattern, which is *why* Chapter 4's discussion of polymorphism matters here — you can swap one algorithm for a completely different one by changing one line:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()      # 1. instantiate the model (with any settings/hyperparameters)
model.fit(X_train, y_train)     # 2. train it on labeled data
predictions = model.predict(X_test)   # 3. predict on new data
```
Some models also support `.fit_transform()` (fit and immediately transform — common for preprocessing steps like scaling) and `.score()` (a quick built-in performance metric).

## 11.8 Preprocessing: Scaling

Many algorithms (especially anything using distance, like KNN, or gradient descent, like linear/logistic regression and neural nets) perform badly if features are on very different scales (e.g. "age" 0-100 vs "income" 0-500,000).

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = StandardScaler()                  # transforms to mean=0, std=1
X_train_scaled = scaler.fit_transform(X_train)   # fit ONLY on training data
X_test_scaled = scaler.transform(X_test)         # then apply the SAME transform to test data
```
**Critical rule: always fit the scaler on training data only, then apply it to the test set.** Fitting on the full dataset (including test data) leaks information from the test set into training — a subtle but common bug called **data leakage**, and it makes your evaluation numbers lie to you.

```python
scaler = MinMaxScaler()   # squashes everything into a fixed range, typically [0, 1]
```

## 11.9 Preprocessing: Encoding Categorical Data

Models need numbers, not text categories.

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd

# One-hot encoding: each category becomes its own 0/1 column — use for UNORDERED categories
df_encoded = pd.get_dummies(df, columns=["city"], drop_first=True)

# Label encoding: assigns 0,1,2... — use ONLY for ORDERED categories (e.g. "low","medium","high")
encoder = LabelEncoder()
df["size_encoded"] = encoder.fit_transform(df["size"])
```
Using label encoding on an *unordered* category (like city names) is a common beginner mistake — it implicitly tells the model "Chicago (1) is between NYC (0) and LA (2)," a relationship that doesn't actually exist. One-hot encoding avoids inventing a false order.

## 11.10 Baselines: The Model You Must Beat

**Intuition:** before celebrating "85% accuracy," ask — 85% compared to what? A **baseline** is the simplest possible model, and it's the number everything else has to beat to count as real progress.

```python
from sklearn.dummy import DummyClassifier, DummyRegressor

baseline = DummyClassifier(strategy="most_frequent")   # always predicts the most common class
baseline.fit(X_train, y_train)
baseline.score(X_test, y_test)   # e.g. 0.94 on a heavily imbalanced dataset
```
**Real world:** on a fraud dataset where 94% of transactions are legitimate, a baseline that always predicts "not fraud" scores 94% accuracy while catching zero fraud. If your trained model scores 95%, you've barely beaten a model that does nothing useful at all — a sign to look at a better metric (Chapter 12.11) rather than celebrate. Always compute a baseline first; it recalibrates what "good" actually means for your specific dataset.

## 11.11 A Complete First Model, End to End

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load
df = pd.read_csv("house_prices.csv")

# 2. Preprocess
df = df.dropna()
X = df[["square_feet", "bedrooms", "bathrooms"]]
y = df["price"]

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Baseline, then the real model
baseline = DummyRegressor(strategy="mean").fit(X_train_scaled, y_train)
print("Baseline R²:", baseline.score(X_test_scaled, y_test))     # e.g. 0.0

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 6. Predict & Evaluate
predictions = model.predict(X_test_scaled)
print("RMSE:", mean_squared_error(y_test, predictions, squared=False))
print("R²:", r2_score(y_test, predictions))                       # compare against the baseline above
```

## 11.12 When Models Fail: A Field Guide

Every real ML project runs into some subset of these. Recognizing the symptom is most of the fix:

| Symptom | Likely cause | Where it's covered |
|---|---|---|
| Great train score, poor test score | Overfitting | 11.5 |
| Poor score on both train and test | Underfitting | 11.5 |
| Suspiciously perfect test score | Data leakage | 11.8, 12.14 |
| High accuracy but useless predictions | Class imbalance + wrong metric | 11.10, 12.11 |
| Score varies wildly across CV folds | Too little data, or high variance | 11.6, 12.12 |
| Good offline score, bad in production | Data drift, or a leaked feature not available at inference time | Chapter 14.6 |
| Model plateaus no matter what you tune | Poor or insufficient features, not a model problem | 8.11, 11.3 |

**Common mistake:** reaching for a fancier model (Random Forest → Gradient Boosting → a neural network) when the actual problem is one of the rows in this table — usually leakage, imbalance, or weak features. A better model trained on the same flawed setup just overfits the flaw more confidently.

---

## What You Learned

- The full real-world ML workflow, from problem statement to monitoring
- Train/validation/test splitting, and why two-way splits break down under tuning
- Overfitting and underfitting, including what the gap actually looks like in numbers
- Bias vs. variance as the formal framing behind both
- Scaling, encoding, baselines, and a field guide to common failure modes

## Common Mistakes

- Tuning against the test set instead of a validation set, silently overfitting to it
- Trusting accuracy alone on an imbalanced dataset
- Skipping a baseline model and having no real reference point for "is this actually good?"

## Quick Check

1. Why does repeatedly checking test-set performance while tuning quietly break the point of a test set?
2. A model has 99% train accuracy and 65% test accuracy — is this high bias or high variance?
3. Why might a fraud-detection model with 99% accuracy still be useless?

## Practice

1. Load any classification dataset, fit a `DummyClassifier` baseline, then fit a real model and report the improvement.
2. Deliberately overfit a `DecisionTreeClassifier` (no `max_depth` limit) and print the train/test accuracy gap.
3. Split a dataset into train/validation/test (60/20/20) and confirm the sizes with `.shape`.

## Challenge

Take the overfitting example in 11.5 and fix it two different ways — first by constraining `max_depth`, then by switching to a `RandomForestClassifier` — and compare the train/test gap for all three versions side by side.

## Where Next?

**Next: Chapter 12 goes deep on the actual algorithms — how regression, classification, and clustering models work, from intuition through to their common failure modes.**
