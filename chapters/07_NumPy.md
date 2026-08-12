# Chapter 7: NumPy — The Foundation of Python ML

```bash
pip install numpy
```
```python
import numpy as np      # "np" is the near-universal convention
```

## 7.1 Why NumPy Exists

A plain Python list stores each element as a full Python object, with its own type info and memory overhead, scattered across memory. A NumPy **array** (`ndarray`) stores raw numbers of a single type in one contiguous memory block, and math operations run as compiled C loops instead of Python bytecode.

**The result: NumPy operations are typically 10-100x faster than equivalent pure-Python loops.** Every ML library — Pandas, Scikit-learn, TensorFlow, PyTorch — represents data as NumPy arrays (or something directly modeled on them) internally. This is the single most important library to understand deeply before moving further.

## 7.2 Creating Arrays

```python
a = np.array([1, 2, 3, 4])              # 1D array, from a list
b = np.array([[1, 2, 3], [4, 5, 6]])    # 2D array (matrix), from a list of lists

np.zeros((3, 4))          # 3x4 array of zeros
np.ones((2, 2))           # 2x2 array of ones
np.full((2, 2), 7)        # 2x2 array filled with 7
np.eye(3)                 # 3x3 identity matrix
np.arange(0, 10, 2)       # [0, 2, 4, 6, 8] — like range(), but returns an array
np.linspace(0, 1, 5)      # [0, 0.25, 0.5, 0.75, 1.0] — 5 evenly spaced points
np.random.rand(3, 3)      # 3x3 array of random floats, uniform [0,1)
np.random.randn(3, 3)     # 3x3 array, standard normal distribution
np.random.randint(0, 10, size=(2,3))   # random integers
```

## 7.3 Array Attributes

```python
a = np.array([[1, 2, 3], [4, 5, 6]])

a.shape       # (2, 3) — (rows, columns)
a.ndim        # 2 — number of dimensions
a.size        # 6 — total number of elements
a.dtype       # dtype('int64') — the data type of every element
```
**Key concept: a NumPy array holds ONE data type for all elements** (unlike a Python list, which can mix types). This uniformity is exactly what makes the fast compiled math possible.

## 7.4 Indexing & Slicing

```python
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

a[0]           # [1, 2, 3] — first row
a[0, 1]        # 2 — row 0, column 1
a[:, 1]        # [2, 5, 8] — every row, column 1 (a column slice)
a[0:2, 0:2]    # [[1,2],[4,5]] — a sub-matrix
a[a > 5]       # [6, 7, 8, 9] — BOOLEAN INDEXING: elements matching a condition
a[a > 5] = 0   # sets all elements greater than 5 to 0, in place
```
Boolean indexing (also called **masking**) is one of the most-used NumPy patterns in real data work: build a `True`/`False` array by evaluating a condition, then use it to filter or modify.

## 7.5 Vectorized Operations

This is NumPy's core idea: apply an operation to an entire array at once, no explicit loop needed.

```python
a = np.array([1, 2, 3, 4])

a + 10        # [11, 12, 13, 14] — adds 10 to every element
a * 2         # [2, 4, 6, 8]
a ** 2        # [1, 4, 9, 16]
a > 2         # [False, False, True, True]

b = np.array([10, 20, 30, 40])
a + b         # [11, 22, 33, 44] — element-wise addition
a * b         # [10, 40, 90, 160] — element-wise multiplication (NOT matrix multiplication)
```

**Broadcasting** — NumPy's rules for combining arrays of different shapes:
```python
a = np.array([[1, 2, 3], [4, 5, 6]])   # shape (2, 3)
b = np.array([10, 20, 30])              # shape (3,)
a + b
# [[11, 22, 33],
#  [14, 25, 36]]
# b is "stretched" across each row without actually copying memory
```
Rule of thumb: two dimensions are compatible if they're equal, or if one of them is 1 (in which case it gets "broadcast" to match).

## 7.6 Matrix Operations (Linear Algebra)

Critical for understanding what's happening inside every neural network layer.

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

A @ B                    # matrix multiplication (also np.matmul(A, B))
A.T                       # transpose — flips rows and columns
np.linalg.inv(A)          # matrix inverse
np.linalg.det(A)          # determinant
np.linalg.eig(A)          # eigenvalues and eigenvectors
np.dot(A, B)              # dot product (same as @ for 2D arrays)
```
**Why this matters:** a "neural network layer" is, mathematically, `output = activation(input @ weights + bias)`. Every forward pass through a network is a sequence of matrix multiplications — this is why GPUs (built for fast parallel matrix math) are what make deep learning practical.

## 7.7 Aggregation Functions

```python
a = np.array([[1, 2, 3], [4, 5, 6]])

a.sum()            # 21 — sum of all elements
a.sum(axis=0)      # [5, 7, 9] — sum DOWN each column
a.sum(axis=1)      # [6, 15]   — sum ACROSS each row
a.mean()           # 3.5
a.std()            # standard deviation
a.min(), a.max()   # 1, 6
a.argmax()         # 5 — INDEX of the max value (flattened)
np.median(a)       # 3.5
```
`axis` trips up almost everyone at first: **`axis=0` collapses rows (operates down each column), `axis=1` collapses columns (operates across each row).**

## 7.8 Reshaping

```python
a = np.arange(12)          # [0, 1, 2, ..., 11]
a.reshape(3, 4)             # reshape into a 3x4 matrix (must have the same total size)
a.reshape(3, -1)            # -1 means "figure this dimension out automatically" → (3, 4)
a.flatten()                  # collapse back to 1D
a.reshape(-1, 1)             # turn a 1D array into a column vector — extremely common before feeding into a model
```

## 7.9 A Worked Example: Normalizing Data

Almost every ML pipeline starts by scaling raw features. Here's the actual math, done manually with NumPy (Scikit-learn will do this for you in Chapter 11, but seeing it raw matters):

```python
data = np.array([10, 20, 30, 40, 50], dtype=float)

# Min-max scaling: squash everything into [0, 1]
normalized = (data - data.min()) / (data.max() - data.min())
# [0. , 0.25, 0.5 , 0.75, 1. ]

# Standardization (z-score): mean 0, standard deviation 1
standardized = (data - data.mean()) / data.std()
```

---

## What You Learned

- Why NumPy arrays are fast: one data type, contiguous memory, compiled math
- Creating, indexing, slicing, and boolean-masking arrays
- Vectorized operations and broadcasting — operating on whole arrays without explicit loops
- Matrix operations (`@`, `.T`, `np.linalg`) and aggregations (`.sum()`, `.mean()`, `axis=`)
- Reshaping, and manual feature scaling with NumPy

## Common Mistakes

- Writing a Python `for` loop over a NumPy array instead of a vectorized operation — usually 10-100x slower, and it defeats the entire point of using NumPy.
- Confusing `axis=0` and `axis=1` — remember: `axis=0` collapses *down* the rows (per-column result), `axis=1` collapses *across* the columns (per-row result).
- Modifying a slice of an array and being surprised the original changed too — NumPy slices are *views*, not copies. Use `.copy()` when you need an independent copy.

## Quick Check

1. Why is `a + b` on two NumPy arrays so much faster than a Python `for` loop doing the same thing?
2. What does `array[array > 5]` do, conceptually?
3. What's the difference between `.reshape(3, 4)` and `.reshape(3, -1)`?

## Practice

1. Create a 5x5 array of random integers between 0 and 100, then find the mean of each row and each column.
2. Given an array of temperatures in Celsius, convert it to Fahrenheit in one vectorized line (`F = C * 9/5 + 32`).
3. Use boolean indexing to replace every negative number in an array with 0.

## Challenge

Implement min-max normalization and z-score standardization as two small functions, then verify they produce the same results as `sklearn.preprocessing.MinMaxScaler`/`StandardScaler` on a small test array (you'll meet those classes properly in Chapter 11).

## Where Next?

**Next: Chapter 8 covers Pandas — built on top of NumPy, and the primary tool for loading, cleaning, and exploring real-world tabular data.**
