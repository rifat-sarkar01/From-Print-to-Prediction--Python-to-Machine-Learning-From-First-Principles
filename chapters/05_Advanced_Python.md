# Chapter 5: Advanced Python

## 5.1 Exception Handling

```python
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"Error: {e}")
except (TypeError, ValueError) as e:    # catch multiple exception types
    print(f"Bad input: {e}")
else:
    print("No error occurred")          # runs only if try succeeded
finally:
    print("This always runs")           # cleanup — runs no matter what
```

**Raising your own exceptions:**
```python
def withdraw(balance, amount):
    if amount > balance:
        raise ValueError("Insufficient funds")
    return balance - amount

try:
    withdraw(100, 150)
except ValueError as e:
    print(e)   # Insufficient funds
```

**Custom exception classes** (common in larger ML pipelines to distinguish error types):
```python
class DataValidationError(Exception):
    pass

def load_data(path):
    if not path.endswith(".csv"):
        raise DataValidationError(f"{path} is not a CSV file")
```

## 5.2 Generators

A **generator** produces values one at a time, on demand, instead of building the whole sequence in memory at once. Critical for ML when working with datasets too large to fit in RAM.

```python
def count_up_to(n):
    i = 1
    while i <= n:
        yield i      # PAUSES here and returns i; resumes from here on the next call
        i += 1

for num in count_up_to(5):
    print(num)       # 1, 2, 3, 4, 5 — one at a time

gen = count_up_to(3)
next(gen)   # 1
next(gen)   # 2
next(gen)   # 3
# next(gen) # raises StopIteration — exhausted
```
Compare with a **generator expression** (like a list comprehension, but lazy — computes values one at a time instead of all at once):
```python
squares = (x**2 for x in range(1000000))   # instant — nothing computed yet
sum(squares)                                # only NOW does it compute, one at a time
```
This is why libraries like `tf.data` and PyTorch's `DataLoader` are built around generator-like patterns — you can stream a 100GB dataset through a model without ever loading it all into memory.

## 5.3 Decorators

A **decorator** wraps a function to add behavior without modifying the function's own code.

```python
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.4f}s")
        return result
    return wrapper

@timer                       # equivalent to: slow_function = timer(slow_function)
def slow_function():
    time.sleep(1)

slow_function()   # prints: slow_function took 1.0002s
```
Read `@timer` as "pass this function into `timer`, and use whatever comes back instead." You'll use decorators constantly in ML code: `@tf.function` (compiles a Python function into a fast TensorFlow graph), `@app.route(...)` in Flask when deploying a model as an API, `@staticmethod` and `@property` (below).

```python
class Circle:
    def __init__(self, radius):
        self.radius = radius

    @property                       # lets you call .area like an attribute, not area()
    def area(self):
        return 3.14159 * self.radius ** 2

    @staticmethod                   # doesn't need self — a utility function attached to the class
    def describe():
        return "A round shape"

c = Circle(5)
print(c.area)          # 78.53975 — no parentheses needed
print(Circle.describe())
```

## 5.4 Context Managers (`with`)

Guarantees setup/cleanup code runs, even if an error occurs — most commonly seen opening files:

```python
with open("data.txt", "r") as f:
    contents = f.read()
# file is automatically closed here, even if an error happened above
```
Writing your own:
```python
class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print(f"Elapsed: {time.time() - self.start:.4f}s")

with Timer():
    time.sleep(1)
```
You'll see this pattern in ML code for things like `with tf.GradientTape() as tape:` (records operations for automatic differentiation) or `with torch.no_grad():` (temporarily disables gradient tracking during inference).

## 5.5 Iterators and the Iterator Protocol

Anything you can put in a `for` loop is an **iterable**. Under the hood:

```python
class Counter:
    def __init__(self, limit):
        self.limit = limit
        self.count = 0

    def __iter__(self):        # returns the iterator object itself
        return self

    def __next__(self):        # returns the next value, or raises StopIteration
        if self.count < self.limit:
            self.count += 1
            return self.count
        raise StopIteration

for num in Counter(3):
    print(num)    # 1, 2, 3
```
This is exactly the machinery a `for` loop relies on for *any* object — lists, dicts, files, generators, and PyTorch `DataLoader`s all implement `__iter__`/`__next__` (or are built from things that do).

## 5.6 Type Hints

Optional annotations that document expected types. Python doesn't enforce them at runtime, but editors and tools like `mypy` use them to catch bugs early — standard practice in professional ML codebases.

```python
def add(a: int, b: int) -> int:
    return a + b

from typing import List, Dict, Optional

def process(names: List[str], scores: Dict[str, float]) -> Optional[float]:
    if not scores:
        return None
    return sum(scores.values()) / len(scores)
```

## 5.7 The Global Interpreter Lock (GIL) — Why It Matters for ML

Python has a **GIL**: only one thread executes Python bytecode at a time, even on a multi-core machine. This means plain Python `threading` doesn't speed up CPU-heavy work (like training a model). Two ways around it, both used constantly in ML:

- **`multiprocessing`** — runs separate Python processes, each with its own GIL, for true CPU parallelism.
- **Releasing the GIL in C** — libraries like NumPy, TensorFlow, and PyTorch do their heavy math in optimized C/C++/CUDA code that releases the GIL, so the actual number-crunching *does* run in parallel even though it's "called from" Python.

This is the real answer to "isn't Python slow for machine learning?" — the Python you write is mostly a thin, readable control layer sitting on top of highly optimized compiled code doing the actual work.

---

## What You Learned

- `try`/`except`/`else`/`finally`, and writing custom exception classes
- Generators and `yield` — for processing data too large to fit in memory
- Decorators, `@property`, and `@staticmethod`
- Context managers (`with`) and the `__enter__`/`__exit__` protocol
- Why NumPy releases the GIL, and why that's the real answer to "isn't Python slow?"

## Code to Remember

```python
try:
    risky()
except SpecificError as e:
    handle(e)
finally:
    cleanup()

def gen():
    yield value            # pauses here, resumes on next()

@decorator
def f(): ...

with resource() as r:
    use(r)                 # cleanup guaranteed even on error
```

## Common Mistakes

- Catching a bare `except:` (catches *everything*, including typos and `KeyboardInterrupt`) instead of the specific exception you expect.
- Building a huge list in memory when a generator would do — a common source of "why is this using so much RAM" bugs on large datasets.
- Forgetting that a generator is exhausted after one full pass — you can't loop over it twice without recreating it.

## Quick Check

1. Why would you use a generator instead of returning a list?
2. What problem do decorators solve that you couldn't do by just editing the function directly?
3. What does a context manager guarantee that a plain `open()`/`close()` pair doesn't?

## Practice

1. Write a generator function `even_numbers(n)` that yields even numbers up to `n`.
2. Write a decorator `@log_calls` that prints a function's name every time it's called.
3. Write a custom exception `InsufficientFundsError` and raise it from a `withdraw()` function.

## Challenge

Write your own context manager (as a class, with `__enter__`/`__exit__`) that temporarily changes the current working directory and restores it afterward, even if an error occurs inside the `with` block.

## Where Next?

**Next: Chapter 6 covers file handling and the parts of Python's standard library you'll use constantly: `os`, `datetime`, `json`, `re`, and `itertools`.**
