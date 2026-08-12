# Chapter 10: Mathematics for Machine Learning

Here's the deal: you don't need a math degree to do machine learning. You need about a dozen ideas, understood well, that show up again and again. This chapter covers exactly those — nothing more.

Each idea follows the same path: **intuition first, formula second, NumPy third.** If you already know some of this, skim it — it's written to be a reference you come back to, not a wall you have to climb.

## 10.1 Scalars, Vectors, and Matrices

A **scalar** is just a single number: `5`, `3.14`, `-2`. In ML code, a single prediction, a single loss value, a learning rate — all scalars.

A **vector** is an ordered list of numbers — think of it as a point in space, or a single row of features.
```python
import numpy as np
house = np.array([1500, 3, 2])   # [square_feet, bedrooms, bathrooms] — one house, as a vector
```
Geometrically, a vector is an arrow from the origin to that point. A **3-feature house** is a point in 3D space; a **784-pixel image** is a point in 784-dimensional space. You can't picture 784 dimensions — nobody can — but the *math* works identically to the 2D and 3D cases you can picture, which is exactly why linear algebra scales.

A **matrix** is a 2D grid of numbers — many vectors stacked together.
```python
houses = np.array([
    [1500, 3, 2],
    [2100, 4, 3],
    [900,  1, 1],
])   # 3 houses (rows) x 3 features (columns)
houses.shape   # (3, 3)
```
This is precisely what a Pandas DataFrame becomes the moment you feed it to a model: a matrix, conventionally called `X`, where each **row is one example** and each **column is one feature**.

## 10.2 Matrix Operations You'll Actually Use

**Addition and scalar multiplication** work element-by-element — you already saw this in Chapter 7:
```python
a = np.array([1, 2, 3])
a + 10        # [11, 12, 13]
a * 2         # [2, 4, 6]
```

**The dot product** is the one operation worth truly understanding, because it's the atomic unit of every neural network layer and every linear model.

**Intuition:** the dot product multiplies two vectors position-by-position and adds up the results. It answers the question *"how much does vector A point in the same direction as vector B?"* — a similarity score, in numeric form.

```
dot([1, 2, 3], [4, 5, 6]) = (1×4) + (2×5) + (3×6) = 4 + 10 + 18 = 32
```
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
np.dot(a, b)     # 32
a @ b            # 32 — same thing, @ is the dot-product/matmul operator
```

**Matrix multiplication** is just many dot products at once — each output cell is the dot product of one row from the first matrix and one column from the second.
```python
weights = np.array([300, 50000, 10000])   # a "weight" per feature
houses @ weights
# [1500*300 + 3*50000 + 2*10000,   → house 1's weighted score
#  2100*300 + 4*50000 + 3*10000,   → house 2's weighted score
#  900*300  + 1*50000 + 1*10000]   → house 3's weighted score
```

**Under the hood:** this is *exactly* what `LinearRegression.predict()` and a `Dense` neural network layer are doing internally — multiplying an input matrix by a weight matrix. Every "layer" in deep learning, no matter how sophisticated the library makes it look, bottoms out in this operation.

**For matrix multiplication `A @ B` to work, the inner dimensions must match:** `(m, n) @ (n, p) → (m, p)`. This single rule explains most of the shape-mismatch errors you'll hit in NumPy, Scikit-learn, and PyTorch — when a model throws a shape error, this is almost always the rule being violated.

## 10.3 Functions, Visually

A **function** takes an input and produces an output: `f(x) = y`. In ML, a model *is* a function — a (very large) function mapping features to a prediction, `f(X) = ŷ` (read "y-hat," the standard notation for "predicted y").

Two functions show up constantly enough to know by shape:

- **Linear**: `f(x) = wx + b` — a straight line. `w` (weight) controls the slope, `b` (bias) shifts it up or down.
- **Sigmoid**: `f(x) = 1 / (1 + e^-x)` — an S-shaped curve that squashes any real number into the range (0, 1). This is what turns a raw model score into something you can read as a probability.

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

sigmoid(0)     # 0.5   — right at the midpoint
sigmoid(10)    # ~1.0  — strongly positive input → confident "yes"
sigmoid(-10)   # ~0.0  — strongly negative input → confident "no"
```

## 10.4 Basic Probability

**Intuition:** probability is just a number between 0 and 1 describing how likely something is. 0 = never happens, 1 = always happens, 0.5 = coin flip.

A few pieces of vocabulary you'll see throughout the ML chapters:

- **Independent events**: one doesn't affect the other (two separate coin flips).
- **Conditional probability**, written `P(A | B)`: the probability of A, *given that* B already happened. "The probability it rains, given that it's cloudy" is a conditional probability — and higher than the plain probability it rains.
- **A distribution** describes how probability is spread across possible outcomes — covered next.

Naive Bayes (Chapter 12) is a full algorithm built directly out of conditional probability — nothing more exotic than the definition above.

## 10.5 Mean, Variance, and Standard Deviation

You already used these in Chapter 7 and 8 — here's what they actually mean.

**Mean** is the average — the "center of mass" of your data.
```python
data = np.array([10, 20, 30, 40, 50])
data.mean()   # 30.0
```

**Variance** measures how spread out the data is from that center — the average of the *squared* distance from the mean.
```
variance = mean((x - mean(x))^2)
```
**Standard deviation** is just the square root of variance — it matters because it's back in the *original units* (variance of "dollars" is in "dollars squared," which nobody can intuit; standard deviation is back in dollars).
```python
data.var()    # 200.0
data.std()    # 14.14 — "typical" distance from the mean, in the same units as the data
```

**Real world:** this is exactly what `StandardScaler` (Chapter 11) uses to rescale features — it subtracts the mean and divides by the standard deviation, so every feature ends up centered at 0 with a "typical spread" of 1, putting differently-scaled features on equal footing.

## 10.6 Distributions

A **distribution** describes the full shape of your data, not just its center. The one to know by sight is the **normal (Gaussian) distribution** — the classic bell curve, where values cluster near the mean and get rarer the further out you go.

```python
import matplotlib.pyplot as plt

samples = np.random.randn(10000)   # 10,000 draws from a standard normal distribution
plt.hist(samples, bins=50)
plt.title("Standard Normal Distribution (mean=0, std=1)")
plt.show()
```
Many real-world quantities (heights, measurement errors, sums of many small effects) are approximately normal — which is *why* `np.random.randn()` and `StandardScaler`'s "mean 0, std 1" target both center on this specific shape. It's also why checking a numeric feature's histogram (Chapter 9) matters: a heavily skewed feature often benefits from a log transform before it goes into a linear model.

## 10.7 Derivatives and Gradients — The Idea That Makes Training Possible

**Intuition:** a derivative answers *"if I nudge the input a tiny bit, how much does the output change, and in which direction?"* It's the slope of a function at a single point.

```
f(x) = x²
f'(x) = 2x     ← the derivative: at any point x, the slope of x² is 2x
```
At `x = 3`, the slope is `6` — meaning if you increase `x` slightly, `f(x)` increases about 6 times as fast. At `x = -3`, the slope is `-6` — increasing `x` *decreases* `f(x)`.

**This single fact is the entire engine behind training a model.** If you know the slope of your error with respect to a weight, you know exactly which direction to move that weight to make the error smaller.

A **gradient** is just the derivative generalized to a function with many inputs (like a model with thousands of weights) — a vector of "how much does the output change per input," one entry per input. Chapter 12's math sections give the exact gradient formulas for specific models; here, you only need the concept.

```python
def f(x):
    return x ** 2

def numerical_derivative(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)   # the slope, estimated numerically

numerical_derivative(f, 3)    # ≈ 6.0 — matches the algebraic answer, 2*3
```
**Under the hood:** this is a *numerical* approximation, useful for building intuition. Real frameworks (PyTorch, TensorFlow) use **automatic differentiation** — computing exact derivatives via the chain rule, applied mechanically through every operation in your model. That machinery is what `loss.backward()` (Chapter 13) is actually doing.

## 10.8 Loss Functions

A **loss function** turns "how wrong was this prediction?" into a single number — the quantity every model is trying to minimize during training. You met two of these already, without the formal name:

```python
import numpy as np

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)     # regression: penalizes big misses heavily (squared)

def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)      # avoid log(0)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
```
**Common mistake:** picking the wrong loss for the problem — e.g. using mean squared error on a classification task. MSE assumes the target is a continuous number; on probabilities it produces a poorly-shaped surface to optimize. This is exactly why Chapter 12 pairs specific losses with specific problem types (`mse` for regression, `binary_crossentropy` for two-class classification, `categorical_crossentropy` for multi-class).

## 10.9 Gradient Descent, End to End

Now put it together: gradient descent uses the gradient (10.7) of the loss (10.8) with respect to the model's weights to repeatedly nudge those weights in the direction that reduces error.

```
new_weight = old_weight - learning_rate × gradient
```
- If the gradient is positive (loss increases as the weight increases), we *decrease* the weight.
- If the gradient is negative, we *increase* it.
- The **learning rate** controls the step size — too large and training overshoots and diverges; too small and training crawls.

**Experiment:** here's the entire idea, fitting a line to data with nothing but NumPy and a loop — no Scikit-learn, no TensorFlow:

```python
import numpy as np

# y = 3x + 7, plus a little noise
X = np.linspace(0, 10, 50)
y = 3 * X + 7 + np.random.randn(50)

w, b = 0.0, 0.0          # start with a bad guess
learning_rate = 0.01

for epoch in range(1000):
    y_pred = w * X + b
    error = y_pred - y

    # gradients of mean squared error with respect to w and b (calculus, worked out in advance)
    grad_w = (2 / len(X)) * np.sum(error * X)
    grad_b = (2 / len(X)) * np.sum(error)

    w -= learning_rate * grad_w
    b -= learning_rate * grad_b

print(f"Learned: y = {w:.2f}x + {b:.2f}")   # should land close to y = 3x + 7
```
Try changing `learning_rate` to `0.5` and watch it diverge (`w`/`b` explode toward infinity or `nan`), or to `0.0001` and watch it barely move after 1000 epochs. This is the single most useful experiment in the whole book for building real intuition for training — every optimizer in Chapter 13 (`Adam`, `SGD`) is a more sophisticated variation on exactly this loop.

## 10.10 How This Maps to the Rest of the Book

| Math idea | Where it resurfaces |
|---|---|
| Dot product / matrix multiplication | Every model's `.predict()`, every neural network layer |
| Mean, variance, standard deviation | `StandardScaler`, feature scaling, EDA |
| Distributions | Data visualization, understanding skew, `np.random` |
| Sigmoid | Logistic regression, binary classification output layers |
| Derivatives / gradients | Backpropagation, every optimizer |
| Loss functions | The single number every `.fit()` call is minimizing |
| Gradient descent | Linear/logistic regression, every neural network |

You now have the full toolkit the rest of this book leans on. From here, the math sections in each algorithm are short — because the hard conceptual work is already done.

---

## What You Learned

- Scalars, vectors, and matrices, and how a DataFrame becomes a matrix the moment a model sees it
- The dot product as a similarity/weighted-sum operation, and why shape mismatches happen
- Sigmoid, for turning raw scores into probabilities
- Mean, variance, standard deviation, and the normal distribution
- Derivatives and gradients as "which way, and how much, to nudge a value"
- Loss functions as the number training is trying to shrink
- Gradient descent, implemented from scratch in about 15 lines

## Common Mistakes

- Treating "the math chapter" as optional — the shape-mismatch errors and "why isn't my model learning" questions in later chapters almost always trace back to one of these ideas.
- Confusing variance and standard deviation — remember: standard deviation is in the original units, variance is not.
- Assuming a bigger learning rate always trains faster — past a certain point it makes training *worse*, not better.

## Quick Check

1. Why must the inner dimensions of two matrices match for `A @ B` to work?
2. What does a derivative of `0` at a point tell you about the function there?
3. Why does logistic regression use cross-entropy loss instead of mean squared error?

## Practice

1. Compute the dot product of `[2, 0, -1]` and `[1, 3, 4]` by hand, then check it with `np.dot`.
2. Generate 1,000 samples from `np.random.randn()`, plot a histogram, and compute the mean and standard deviation of your sample. How close are they to 0 and 1?
3. Modify the gradient descent example in 10.9 to fit `y = -2x + 5` instead of `y = 3x + 7`.

## Challenge

Extend the from-scratch gradient descent example to two features instead of one (i.e. fit `y = w1*x1 + w2*x2 + b`). You'll need a gradient for each weight — the pattern from 10.9 generalizes directly.

## Where Next?

Chapter 11 puts this math to work — the full machine learning workflow, from raw data to a trained, evaluated model, using exactly the concepts from this chapter.
