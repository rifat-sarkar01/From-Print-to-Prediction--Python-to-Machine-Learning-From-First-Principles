# Chapter 3: Functions & Modules

## 3.1 Defining Functions

```python
def greet(name):
    """Return a greeting for the given name. (This is a docstring.)"""
    return f"Hello, {name}!"

message = greet("Alice")
print(message)   # Hello, Alice!
```
- `def` starts a function definition.
- The **docstring** (triple-quoted string right after `def`) documents what the function does — tools and editors display it as help text via `help(greet)`.
- `return` sends a value back to wherever the function was called. Without it, the function returns `None`.

## 3.2 Parameters, Defaults, and Keyword Arguments

```python
def power(base, exponent=2):        # exponent has a DEFAULT value
    return base ** exponent

power(5)             # 25   — uses default exponent=2
power(5, 3)           # 125  — positional argument
power(base=5, exponent=3)   # 125  — keyword argument, order doesn't matter
```

### `*args` and `**kwargs`

```python
def total(*args):                # collects any number of positional args into a tuple
    return sum(args)

total(1, 2, 3, 4)     # 10

def describe(**kwargs):          # collects any number of keyword args into a dict
    for key, value in kwargs.items():
        print(f"{key}: {value}")

describe(name="Alice", age=30)
```
`*args` and `**kwargs` are conventional names (you could call them anything) — the `*` and `**` are what actually matter: `*` gathers extra positional arguments, `**` gathers extra keyword arguments. You'll see these constantly in ML library code, e.g. `model.fit(X, y, **fit_params)`.

## 3.3 Scope

```python
x = 10          # global scope — visible everywhere in the file

def modify():
    x = 20      # this creates a NEW local variable x, shadowing the global one
    print(x)    # 20

modify()
print(x)        # still 10 — the global x was untouched

def modify_global():
    global x
    x = 20      # this DOES change the global x

modify_global()
print(x)        # 20
```
Python looks up a name using the **LEGB rule**: Local → Enclosing → Global → Built-in, in that order.

## 3.4 Lambda Functions

Small, anonymous, one-expression functions:
```python
square = lambda x: x ** 2
square(5)   # 25

# most common real use: as a quick key/sort function
people = [("Alice", 30), ("Bob", 25)]
people.sort(key=lambda person: person[1])   # sort by age
```
Anything a lambda can do, a regular `def` function can also do — lambdas are just shorthand for simple, throwaway logic.

## 3.5 Higher-Order Functions

Functions that take other functions as arguments — a core functional-programming pattern in Python.

```python
nums = [1, 2, 3, 4, 5]

list(map(lambda x: x * 2, nums))          # [2, 4, 6, 8, 10] — apply to every element
list(filter(lambda x: x % 2 == 0, nums))  # [2, 4]           — keep matching elements

from functools import reduce
reduce(lambda a, b: a + b, nums)          # 15 — cumulative reduction to a single value
```
In practice, list comprehensions (`[x*2 for x in nums]`) are usually preferred over `map`/`filter` in modern Python for readability — but you'll see all of these in the wild.

## 3.6 Modules and Imports

A **module** is just a `.py` file. A **package** is a folder of modules with an `__init__.py` file.

```python
import math
math.sqrt(16)          # 4.0

import math as m       # aliasing
m.sqrt(16)

from math import sqrt, pi     # import specific names directly
sqrt(16)

from math import *     # imports everything — generally discouraged, pollutes namespace
```

### Writing your own module

**`mymath.py`**
```python
def add(a, b):
    return a + b
```

**`main.py`**
```python
import mymath
mymath.add(2, 3)   # 5
```

### The `if __name__ == "__main__":` idiom

```python
def main():
    print("Running as a script")

if __name__ == "__main__":
    main()
```
Every Python file has a `__name__` variable. If the file is run directly, `__name__` equals `"__main__"`. If the file is *imported* by another file, `__name__` equals the module's name instead. This lets a file be both a reusable module *and* a runnable script, without the "runnable script" part firing off accidentally when someone just wants to import a function from it.

## 3.7 Virtual Environments (Why They Matter for ML)

Different projects often need different, conflicting versions of the same library. A **virtual environment** is an isolated Python installation per project.

```bash
python3 -m venv myenv          # create it
source myenv/bin/activate      # activate it (Mac/Linux)
myenv\Scripts\activate         # activate it (Windows)
pip install numpy pandas       # installs ONLY inside this environment
deactivate                     # exit it
```
This matters enormously in machine learning work, where library version mismatches (e.g. a model saved with one version of TensorFlow failing to load in another) are one of the most common real-world headaches.

## Mini Project: CLI Calculator

Everything you need for this is already in Chapters 1-3 — no new syntax, just putting it together.

**Goal:** a command-line calculator that loops, asking for two numbers and an operator, until the user types `"quit"`.

**Steps:**

1. Write four small functions: `add(a, b)`, `subtract(a, b)`, `multiply(a, b)`, `divide(a, b)` — handle division by zero with a clear message instead of crashing.
2. Write a `calculate(a, b, op)` function that dispatches to the right one based on `op` (a dictionary mapping `"+"`, `"-"`, `"*"`, `"/"` to functions works nicely — a preview of why functions being "first-class values" in Python is useful).
3. Wrap it in a `while True:` loop that reads input, breaks on `"quit"`, and otherwise prints the result.
4. **Stretch:** keep a running history list of past calculations and let the user type `"history"` to see it.

This is a genuinely useful pattern — a dictionary of functions instead of a long `if/elif` chain — that you'll see again when you build model-selection code in the ML chapters.

---

## What You Learned

- Defining functions with `def`, default arguments, and docstrings
- `*args` and `**kwargs` for flexible function signatures
- Scope and the `global` keyword
- Lambda functions, `map`/`filter`/`reduce`
- Modules, imports, the `if __name__ == "__main__":` idiom, and virtual environments

## Code to Remember

```python
def f(a, b=10, *args, **kwargs): ...     # the full parameter pattern, in order
if __name__ == "__main__":
    main()
```

## Common Mistakes

- Using a mutable default argument (`def f(items=[])`) — that list is created *once*, when the function is defined, and reused across every call, silently accumulating state. Use `items=None` and set `items = items or []` inside the function instead.
- Shadowing a global variable by accident — assigning to a name inside a function creates a *local* variable unless you explicitly declare `global`.
- Overusing lambdas for anything beyond a one-line expression, hurting readability for no real benefit over a named function.

## Quick Check

1. What's the difference between `*args` and `**kwargs`?
2. Why does `def f(items=[])` behave unexpectedly across repeated calls?
3. What does `if __name__ == "__main__":` actually check?

## Practice

1. Write a function `describe(**kwargs)` that prints each keyword argument as `"key: value"`.
2. Write a decorator-free function `apply_twice(f, x)` that applies function `f` to `x` twice, and test it with a lambda that squares a number.
3. Turn one of your Chapter 1 scripts into a proper module with a `main()` function guarded by `if __name__ == "__main__":`.

## Challenge

Extend the CLI Calculator above to support a `"history"` command, and a way to redo the last calculation with different numbers.

## Where Next?

**Next: Chapter 4 covers Object-Oriented Programming — how Python lets you build your own custom data types.**
