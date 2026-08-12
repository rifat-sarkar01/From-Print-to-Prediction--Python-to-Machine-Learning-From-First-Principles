# Chapter 12: Machine Learning Algorithms Deep Dive

Every algorithm in this chapter follows the same path: **intuition → how it works → the math → Scikit-learn → when to use it → where it breaks.** A few also get a from-scratch NumPy implementation, where actually building it teaches you something the library call can't.

## 12.1 Linear Regression

**Intuition:** you're drawing the straight line through a scatter plot that comes closest to every point at once — not passing through them, just minimizing the total distance.

**How it works:** the model learns one weight per feature and a bias term, then predicts a weighted sum. Training searches for the weights that minimize the total squared error across every training example — a method called **Ordinary Least Squares**, typically solved with gradient descent (Chapter 10.9).

**Math:** `y = w1*x1 + w2*x2 + ... + wn*xn + b` — exactly the dot product from Chapter 10.2, plus a bias.

**From scratch:** you already built this in Chapter 10.9 — gradient descent on a line, in about 15 lines of NumPy. That *is* linear regression; Scikit-learn just solves it more efficiently.

**Scikit-learn:**
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

model.coef_          # the learned weight for each feature
model.intercept_     # the learned bias term
```

**When to use:** the relationship between features and target is roughly linear, and you want an interpretable model — each coefficient directly tells you "for every 1-unit increase in this feature, the prediction changes by this much."

**Limitations:** assumes a linear relationship; sensitive to outliers (a single extreme point can pull the whole line); struggles when features are highly correlated with each other (multicollinearity).

**Common mistake:** interpreting a large coefficient as "this feature matters most" without checking that features were scaled first (Chapter 11.8) — on unscaled data, coefficient size just reflects the feature's units, not its importance.

**Mini experiment:** add a single extreme outlier to a small synthetic dataset and refit — watch how much the line moves. Then try `Ridge` below on the same data and compare.

**Regularized variants**, which penalize overly large weights to reduce overfitting:
```python
from sklearn.linear_model import Ridge, Lasso

Ridge(alpha=1.0)   # shrinks weights smoothly (L2 penalty) — good default when most features matter
Lasso(alpha=1.0)   # can shrink weights to EXACTLY zero (L1 penalty) — effectively does feature selection
```

## 12.2 Logistic Regression

**Intuition:** despite the name, this is a **classification** algorithm — it draws a boundary between classes rather than a line through numbers, and reports *how confident* it is as a probability.

**How it works:** compute the same weighted sum as linear regression, then squash it through the sigmoid function (Chapter 10.3) to get a number between 0 and 1. Above 0.5 (by default) predicts one class, below predicts the other.

**Math:** `p = sigmoid(w1*x1 + ... + wn*xn + b)`, trained by minimizing cross-entropy loss (Chapter 10.8) instead of squared error.

**From scratch:**
```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(X, y, lr=0.1, epochs=1000):
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(epochs):
        z = X @ w + b
        preds = sigmoid(z)
        error = preds - y
        w -= lr * (X.T @ error) / len(y)     # same gradient-descent shape as Chapter 10.9
        b -= lr * np.mean(error)
    return w, b
```

**Scikit-learn:**
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

model.predict(X_test)              # class predictions: 0 or 1
model.predict_proba(X_test)        # actual probabilities for each class
```

**When to use:** a fast, interpretable baseline for binary or multi-class classification — reach for this before anything fancier.

**Limitations:** assumes a roughly linear decision boundary between classes; like linear regression, sensitive to unscaled features and outliers.

**Common mistake:** treating `.predict()`'s 0/1 output as the whole story and ignoring `.predict_proba()` — a prediction of "1" at 51% confidence and one at 99% confidence are very different in practice, especially when false positives are costly.

## 12.3 Decision Trees

**Intuition:** a flowchart of yes/no questions — "is age > 30? is income > $50k?" — that ends in a prediction.

**How it works:** at each step, the tree picks the feature and threshold that best separates the classes (measured by **Gini impurity** or **entropy** — both just ways of scoring "how mixed are the classes in this group?"), splits the data, and repeats on each half.

**Scikit-learn:**
```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

model = DecisionTreeClassifier(max_depth=5)   # limit depth to prevent overfitting
model.fit(X_train, y_train)

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plot_tree(model, feature_names=X.columns, filled=True)
plt.show()
```

**When to use:** you need a model a non-technical stakeholder can literally read, or a quick baseline that handles non-linear patterns without any preprocessing.

**Limitations:** highly interpretable, but prone to overfitting if left unconstrained — `max_depth`, `min_samples_split`, and `min_samples_leaf` are the key controls.

**Common mistake:** leaving `max_depth=None` (the default) on a tree of any real size, then being surprised by the exact overfitting pattern from Chapter 11.5 — a lone decision tree is the textbook example of high variance.

**Mini experiment:** train the same tree with `max_depth=3`, `max_depth=10`, and `max_depth=None`, and plot train vs. test accuracy for each — you're tracing the bias-variance tradeoff from Chapter 11.6 directly.

## 12.4 Random Forests

**Intuition:** ask a hundred slightly-different decision trees and take a vote, instead of trusting one tree's opinion.

**How it works:** trains many decision trees, each on a random subset of the data (rows) and a random subset of features (columns) at each split, then averages (regression) or votes (classification) across all of them.

**Under the hood:** a single tree can overfit to noise, but many trees trained on different random slices tend to make *different* mistakes — averaging cancels much of that noise out. This "wisdom of crowds" effect (formally: **bagging**, short for bootstrap aggregating) is one of the most reliable improvements in all of ML, and it's why a Random Forest almost always beats its individual trees.

**Scikit-learn:**
```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

model.feature_importances_    # which features mattered most, averaged across all trees
```

**When to use:** a strong, low-effort default for tabular data — good accuracy with far less overfitting risk than a single tree, and little preprocessing required (no scaling needed, unlike KNN or logistic regression).

**Limitations:** slower to train and predict than a single tree; less interpretable (you can't read 100 trees); can still overfit with too many very deep trees on small, noisy data.

**Common mistake:** forgetting that `feature_importances_` reflects *how often and how usefully* a feature was split on — not causation. A high-importance feature is predictive, not necessarily a cause of the outcome.

## 12.5 Gradient Boosting (XGBoost / LightGBM)

**Intuition:** instead of many independent trees voting (Random Forest), build trees one at a time, where each new tree's entire job is to fix the mistakes the previous trees made.

**How it works:** train a small first tree, look at its errors (residuals), train a second tree specifically to predict those errors, add it in, and repeat — each round nudges the combined prediction closer to correct.

```bash
pip install xgboost lightgbm
```
```python
from xgboost import XGBClassifier

model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6)
model.fit(X_train, y_train)
```

**When to use:** structured/tabular data where you want the best achievable accuracy and are willing to tune more carefully than a Random Forest requires. Gradient-boosted trees (XGBoost, LightGBM, CatBoost) are consistently among the top-performing algorithms on this kind of data, and remain a strong choice even alongside deep learning.

**Limitations:** more hyperparameters to tune than a Random Forest (`learning_rate`, `n_estimators`, `max_depth` all interact); more prone to overfitting if `n_estimators` is too high relative to `learning_rate`; trains sequentially, so it's slower to parallelize than bagging.

**Common mistake:** cranking `n_estimators` up with no early stopping — boosting keeps fitting the training data more closely round after round, and without a validation-based stopping rule, it will eventually start memorizing noise.

## 12.6 Support Vector Machines (SVM)

**Intuition:** among all the possible boundaries that separate two classes, pick the one with the widest possible margin — the boundary that stays as far as possible from the nearest point of *either* class.

**How it works:** the closest points to the boundary are the "support vectors" — they're the only points that actually determine where the boundary sits. Everything else could move or vanish without changing the result.

**Math:** for non-linear boundaries, the **kernel trick** implicitly maps data into a higher-dimensional space where a straight boundary *would* separate the classes, without ever computing that expensive mapping directly.

**Scikit-learn:**
```python
from sklearn.svm import SVC

model = SVC(kernel="rbf", C=1.0)   # "rbf" kernel handles non-linear boundaries
model.fit(X_train, y_train)
```
`C` controls the trade-off between a wider margin and fewer misclassifications on the training data — lower `C` means a wider margin with more tolerance for errors.

**When to use:** smaller-to-medium datasets with a clear margin between classes, especially in high-dimensional spaces (like text classification with thousands of word features).

**Limitations:** doesn't scale well to very large datasets (training time grows quickly with row count); requires feature scaling; the kernel and `C` both need tuning, and the results are less interpretable than a tree or linear model.

## 12.7 Naive Bayes

**Intuition:** a direct application of the conditional probability from Chapter 10.4 — "given the words in this email, what's the probability it's spam?" — assuming, naively, that every feature (word) contributes independently.

**How it works:** for each class, it learns how likely each feature value is to appear, then for a new example, multiplies those likelihoods together (via Bayes' theorem) to get a score per class, and predicts whichever class scores highest.

**Scikit-learn:**
```python
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train, y_train)   # classic use case: spam detection with word-count features
```

**When to use:** text classification (spam filtering, topic tagging, sentiment) with word-count or TF-IDF features (Chapter 14.1) — fast to train, works well even on relatively little data, and a strong baseline before trying anything heavier.

**Limitations:** despite working well in practice, the independence assumption is technically wrong (word choice obviously isn't independent) — it can produce poorly calibrated probabilities even when its final class predictions are accurate.

**Common mistake:** using Naive Bayes on features with strong dependencies and expecting well-calibrated `predict_proba()` output — trust its classifications more than its exact probability numbers.

## 12.8 K-Nearest Neighbors (KNN)

**Intuition:** "you are the average of the people closest to you" — to classify a new point, look at its `k` nearest neighbors in the training data and take a majority vote.

**How it works:** no real "training" happens — KNN just stores the training data. At prediction time, it computes the distance from the new point to every stored point, finds the `k` closest, and votes (classification) or averages (regression).

**From scratch:**
```python
import numpy as np
from collections import Counter

def knn_predict(X_train, y_train, x_new, k=5):
    distances = np.sqrt(((X_train - x_new) ** 2).sum(axis=1))   # Euclidean distance to every point
    nearest_indices = distances.argsort()[:k]                    # indices of the k closest
    nearest_labels = y_train[nearest_indices]
    return Counter(nearest_labels).most_common(1)[0][0]          # majority vote
```

**Scikit-learn:**
```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
```

**When to use:** small-to-medium datasets where the decision boundary is irregular and you'd rather not assume any particular shape (unlike linear/logistic regression).

**Limitations:** slow at prediction time on large datasets (it compares against every stored point); the curse of dimensionality — distance becomes less meaningful as feature count grows; **requires feature scaling**, non-negotiably.

**Common mistake:** skipping feature scaling (Chapter 11.8). An unscaled "income" feature (0-500,000) completely dominates the distance calculation over "age" (0-100), making age effectively invisible to the model.

**Mini experiment:** try `n_neighbors=1` vs. `n_neighbors=50` on the same dataset — `k=1` is maximally flexible (high variance, can overfit to single noisy points); a very large `k` oversmooths toward always predicting the majority class (high bias).

## 12.9 Clustering: K-Means

**Intuition:** given a pile of unlabeled points, find `k` natural groupings by guessing `k` center points and letting them settle into place.

**How it works:** (1) place `k` cluster centers randomly, (2) assign every point to its nearest center, (3) move each center to the average position of its assigned points, (4) repeat steps 2-3 until the centers stop moving.

**From scratch:**
```python
import numpy as np

def kmeans(X, k, n_iters=100):
    centers = X[np.random.choice(len(X), k, replace=False)]      # random starting centers
    for _ in range(n_iters):
        distances = np.array([np.linalg.norm(X - c, axis=1) for c in centers])
        labels = distances.argmin(axis=0)                        # assign each point to nearest center
        new_centers = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        if np.allclose(centers, new_centers):
            break                                                  # centers stopped moving — converged
        centers = new_centers
    return labels, centers
```

**Scikit-learn:**
```python
from sklearn.cluster import KMeans

model = KMeans(n_clusters=3, random_state=42)
model.fit(X)
labels = model.labels_               # which cluster each point belongs to
centers = model.cluster_centers_
```

**Choosing `k` — the elbow method:** plot inertia (within-cluster variance) against different values of `k`, and look for the point where the improvement sharply levels off:
```python
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42).fit(X)
    inertias.append(km.inertia_)

plt.plot(range(1, 11), inertias, marker="o")
plt.xlabel("k"); plt.ylabel("Inertia")
plt.show()
```

**When to use:** exploratory grouping of unlabeled data — customer segmentation, document topics, anomaly detection (points far from every center).

**Limitations:** you have to choose `k` in advance; assumes roughly round, similarly-sized clusters, and struggles with oddly-shaped or very unevenly-sized groups; sensitive to feature scaling and to the random starting centers (Scikit-learn mitigates this by re-running with several random starts automatically).

**Common mistake:** running K-Means on unscaled features — exactly the same distance-domination problem as KNN in 12.8, since K-Means is also entirely distance-based.

## 12.10 Dimensionality Reduction: PCA

**Intuition:** if several of your features are highly correlated, they're partly saying the same thing — PCA finds a smaller set of new features that captures most of the original information with less redundancy.

**How it works:** finds the directions (**principal components**) along which the data varies the most, and re-expresses the data in terms of those directions instead of the original features — the first component captures the most variance, the second the next most (while staying perpendicular to the first), and so on.

**Scikit-learn:**
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)     # always scale features BEFORE PCA
pca.explained_variance_ratio_               # how much variance each component captures
```

**When to use:** visualizing high-dimensional data in 2D/3D; speeding up training on very high-dimensional data; reducing multicollinearity before a linear model.

**Limitations:** the new components are combinations of original features and generally aren't interpretable on their own ("principal component 2" doesn't have a real-world meaning the way "square footage" does); you lose some information by design — check `explained_variance_ratio_` to see how much.

**Common mistake:** applying PCA before scaling — since PCA is variance-based, an unscaled feature with a naturally larger numeric range will dominate the first component regardless of whether it's actually the most informative feature.

## 12.11 Model Evaluation Metrics

**For classification:**
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

accuracy_score(y_test, predictions)     # % of correct predictions overall
precision_score(y_test, predictions)    # of predicted positives, how many were actually positive?
recall_score(y_test, predictions)       # of actual positives, how many did we catch?
f1_score(y_test, predictions)           # harmonic mean of precision and recall
confusion_matrix(y_test, predictions)   # [[TN, FP], [FN, TP]]
print(classification_report(y_test, predictions))   # all of the above, per class
```
**Precision vs. recall trade-off in plain terms:** in cancer screening, missing a real case (low recall) is far worse than a false alarm (low precision) — so you'd tune the model to favor recall. In spam filtering, wrongly flagging a real email as spam (low precision) is more annoying than letting one spam email through — so you'd favor precision. **Accuracy alone is misleading on imbalanced data** — a model that always predicts "not fraud" on a dataset that's 99% not-fraud gets 99% accuracy while being completely useless — this is exactly the baseline problem from Chapter 11.10.

**For regression:**
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mean_absolute_error(y_test, predictions)     # average absolute difference from truth
mean_squared_error(y_test, predictions)      # penalizes large errors more heavily (squared)
r2_score(y_test, predictions)                # proportion of variance explained, 1.0 = perfect
```

## 12.12 Cross-Validation

A single train/test split can be misleading — performance can vary depending on *which* data happened to land in the test set. **K-fold cross-validation** splits the data into `k` chunks, trains on `k-1` of them and tests on the last, and repeats `k` times so every chunk gets used as the test set exactly once — then averages the results.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print(scores.mean(), scores.std())    # average performance, and how much it varies across folds
```
A large `scores.std()` is itself informative — it's a direct readout of the variance side of the bias-variance tradeoff (Chapter 11.6): a model whose score swings wildly across folds is unstable, regardless of what its average looks like.

## 12.13 Hyperparameter Tuning

**Hyperparameters** are settings you choose *before* training (like `max_depth` or `n_estimators`) — as opposed to parameters the model *learns* during training (like the weights in linear regression).

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 15],
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring="accuracy")
grid_search.fit(X_train, y_train)

grid_search.best_params_     # the winning combination
grid_search.best_score_      # its cross-validated score
best_model = grid_search.best_estimator_
```
`GridSearchCV` exhaustively tries every combination — thorough but slow on large grids. `RandomizedSearchCV` samples a fixed number of random combinations instead, often finding a nearly-as-good result far faster.

## 12.14 Building a Pipeline

Chains preprocessing and modeling steps into one object — prevents data leakage (Chapter 11.8) automatically, since every step is refit correctly within each cross-validation fold.

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier())
])

pipeline.fit(X_train, y_train)
pipeline.predict(X_test)
```

## Mini Project: Titanic Survival Classifier

You already cleaned this dataset in Chapter 8.11. Now finish the job, end to end, using this chapter's toolkit.

**Steps:**

1. Reuse the cleaned, encoded Titanic DataFrame from Chapter 8.11.
2. Split into train/test, and fit a `DummyClassifier` baseline (Chapter 11.10) — note the score.
3. Train three real models: `LogisticRegression`, `RandomForestClassifier`, and `XGBClassifier`, all through the same `Pipeline` pattern from 12.14.
4. Compare them with `cross_val_score` (12.12) rather than a single split — report mean and standard deviation for each.
5. For your best model, print a full `classification_report` and a confusion matrix. Which mistakes does it make — false survivors, or false casualties?
6. Use `.feature_importances_` (Random Forest) or `.coef_` (Logistic Regression) to report which features mattered most. Does it match your intuition (class, sex, age were historically decisive)?

**Common mistake to watch for:** fitting `StandardScaler` before the train/test split, or engineering features (like `FamilySize`) using statistics computed from the *full* dataset including test rows — both are data leakage (Chapter 11.8), and both will make your reported accuracy look better than the model will actually perform on new passengers.

---

## What You Learned

- Ten core algorithms — linear/logistic regression, decision trees, random forests, gradient boosting, SVM, Naive Bayes, KNN, K-Means, and PCA — each from intuition through to its failure modes
- Classification and regression metrics, and when accuracy alone is misleading
- Cross-validation, hyperparameter tuning, and building leakage-safe pipelines
- A complete Titanic classifier project comparing multiple models against a baseline

## Common Mistakes

- Picking a model before checking whether the problem is even linearly separable — a two-minute scatter plot often tells you whether logistic regression will struggle
- Comparing models on a single train/test split instead of cross-validation
- Reporting accuracy on an imbalanced dataset without also reporting precision/recall

## Quick Check

1. Why does Random Forest resist overfitting better than a single Decision Tree?
2. Why is feature scaling non-negotiable for KNN and K-Means but irrelevant for Decision Trees and Random Forests?
3. Gradient Boosting and Random Forest both use many trees — what's the core difference in how they combine them?

## Practice

1. Train a Decision Tree and a Random Forest on the same dataset and compare their train/test accuracy gap.
2. Run K-Means with `k` from 1 to 10 on a dataset with 2-3 numeric features, and use the elbow method to pick `k`.
3. Compare Logistic Regression and KNN on the same classification dataset with `cross_val_score`.

## Challenge

Build a small "model bake-off": train Logistic Regression, Random Forest, and Gradient Boosting on the same classification dataset, compare them with 5-fold cross-validation, and pick a winner using F1 score rather than accuracy. Justify the choice of F1 over accuracy for your specific dataset.

## Where Next?

**Next: Chapter 13 moves into deep learning — neural networks, TensorFlow/Keras, PyTorch, and the architectures behind modern image and language models.**
