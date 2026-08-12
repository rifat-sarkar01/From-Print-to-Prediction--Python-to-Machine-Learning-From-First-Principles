# Chapter 1: Python Basics

## 1.1 Your First Program

```python
print("Hello, world!")
```

- `print(...)` is a **function** — a named, reusable block of code that does something when you "call" it (invoke it with parentheses).
- `"Hello, world!"` is a **string** — text data, always wrapped in quotes (single `'...'` or double `"..."`, Python treats them identically).
- Nothing needs a semicolon at the end of a line, no `main()` function is required, and nothing needs to be compiled first. Python is an **interpreted language** — it reads and runs your code line by line.

## 1.2 Indentation Is Not Optional

Most languages use `{ }` to group code. Python uses **whitespace (indentation)** itself:

```python
if 5 > 3:
    print("Five is greater than three")
    print("This line is still inside the if-block")
print("This line is NOT inside the if-block")
```

The standard is **4 spaces** per indentation level. Mixing tabs and spaces will cause errors — most editors handle this automatically.

**Why this matters conceptually:** Python's designers made indentation mandatory so that visual structure and logical structure can never disagree — code that *looks* nested always *is* nested.

## 1.3 Variables

```python
age = 25
name = "Alice"
height = 5.6
is_student = True
```

A **variable** is a name bound to a value. Python is:

- **Dynamically typed** — you don't declare a type; `age` could hold an integer now and a string later (though doing so is usually bad practice).
- The `=` sign is **assignment**, not mathematical equality. `age = 25` means "point the name `age` at the value `25`," not "age equals 25" as a statement of fact.

Naming rules: must start with a letter or underscore, can contain letters/numbers/underscores, is case-sensitive (`Age` ≠ `age`), and can't be a reserved keyword (`if`, `for`, `class`, etc).

Convention (PEP 8, Python's style guide): variables and functions use `snake_case`; classes use `PascalCase`; constants use `ALL_CAPS`.

## 1.4 Core Data Types

```python
whole_number   = 42            # int
decimal_number = 3.14          # float
text           = "hello"       # str
truth_value    = True          # bool (True / False, capitalized)
nothing        = None          # NoneType — represents "no value"
```

Check any variable's type with the built-in `type()` function:
```python
print(type(42))        # <class 'int'>
print(type(3.14))      # <class 'float'>
print(type("hi"))      # <class 'str'>
```

**`None` deserves special attention.** It is Python's explicit "nothing here" — different from `0`, `False`, or an empty string. It's commonly used as a default/placeholder value before something real is assigned, and functions that don't explicitly `return` anything return `None` automatically.

### Type Conversion (Casting)

```python
int("5")        # 5      — string to integer
str(5)          # "5"    — integer to string
float("3.14")   # 3.14   — string to float
int(3.99)       # 3      — float to int TRUNCATES (doesn't round)
bool(0)         # False  — 0, "", None, [], {} are all "falsy"
bool(1)         # True   — nearly everything else is "truthy"
```

## 1.5 Operators

**Arithmetic:**
```python
7 + 3   # 10   addition
7 - 3   # 4    subtraction
7 * 3   # 21   multiplication
7 / 3   # 2.333...  true division — ALWAYS returns a float
7 // 3  # 2    floor division — divides then rounds DOWN
7 % 3   # 1    modulo — the remainder
7 ** 3  # 343  exponentiation
```

**Comparison** (return `True`/`False`):
```python
5 == 5   # equal to
5 != 3   # not equal to
5 > 3    # greater than
5 < 3    # less than
5 >= 5   # greater than or equal
5 <= 3   # less than or equal
```

**Logical:**
```python
True and False   # False — both must be True
True or False    # True  — at least one must be True
not True         # False — flips the boolean
```

**Assignment shorthand:**
```python
x = 5
x += 3   # same as x = x + 3   → 8
x -= 1   # x = x - 1
x *= 2   # x = x * 2
x /= 4   # x = x / 4
```

## 1.6 Strings in Depth

```python
name = "Alice"
greeting = f"Hello, {name}!"        # f-string — embeds variables directly
print(greeting)                     # Hello, Alice!
```

An **f-string** (formatted string literal, `f"..."`) lets you insert any Python expression inside `{ }` — including math, function calls, or formatting specs like `{price:.2f}` (2 decimal places).

Common string operations:
```python
s = "Hello, World"
s.lower()          # "hello, world"
s.upper()          # "HELLO, WORLD"
s.strip()          # removes leading/trailing whitespace
s.replace("l","L") # "HeLLo, WorLd"
s.split(", ")      # ["Hello", "World"]  → splits into a list
len(s)             # 12 — number of characters
s[0]               # "H" — indexing (0-based)
s[-1]              # "d" — negative index counts from the end
s[0:5]             # "Hello" — slicing: [start:stop] (stop is EXCLUSIVE)
s[::-1]            # "dlroW ,olleH" — reversed string (step = -1)
```

Strings are **immutable** — `s[0] = "J"` throws an error. Any "modification" (like `.upper()`) actually returns a brand-new string.

## 1.7 Control Flow

### if / elif / else
```python
temperature = 75

if temperature > 90:
    print("It's hot")
elif temperature > 60:
    print("It's nice")
else:
    print("It's cold")
```
`elif` = "else if." Python checks conditions top to bottom and runs the *first* one that's `True`, then skips the rest.

### while loops
```python
count = 0
while count < 5:
    print(count)
    count += 1     # without this, the loop runs forever
```

### for loops
```python
for i in range(5):        # 0, 1, 2, 3, 4
    print(i)

for letter in "abc":      # loops over each character
    print(letter)

fruits = ["apple", "banana", "cherry"]
for fruit in fruits:      # loops over each element
    print(fruit)
```
`range(start, stop, step)` generates a sequence of numbers lazily (without building a full list in memory) — `range(5)` = 0 to 4, `range(2, 10, 2)` = 2, 4, 6, 8.

### break, continue, pass
```python
for i in range(10):
    if i == 5:
        break        # exits the loop immediately
    if i % 2 == 0:
        continue     # skips to the next iteration
    print(i)

def not_done_yet():
    pass             # placeholder — does nothing, avoids a syntax error
```

## 1.8 Getting User Input

```python
name = input("What's your name? ")   # input() always returns a string
age = int(input("What's your age? "))  # cast to int if you need a number
print(f"Hi {name}, you are {age} years old.")
```

## 1.9 Comments

```python
# This is a single-line comment
"""
This is a multi-line string,
often used as a block comment
or as a docstring (documentation) for functions/classes.
"""
```

---

## What You Learned

- Variables, dynamic typing, and Python's core data types (`int`, `float`, `str`, `bool`, `None`)
- Arithmetic, comparison, and logical operators — including the `//`, `%`, and `**` operators
- f-strings and the most common string operations
- `if`/`elif`/`else`, `while`, and `for` loops, plus `break`/`continue`/`pass`

## Code to Remember

```python
for i in range(len(items)):        # avoid this when you can...
    print(items[i])

for item in items:                 # ...prefer this — more Pythonic, less error-prone
    print(item)

x = 5 if condition else 10         # the ternary/conditional expression
```

## Common Mistakes

- Using `=` (assignment) when you mean `==` (comparison) — `if x = 5:` is a syntax error in Python, but the habit is worth breaking early.
- Forgetting that `/` always returns a float, even `10 / 2` — use `//` if you specifically want an integer result.
- Mixing tabs and spaces for indentation, which throws a confusing `TabError`.

## Quick Check

1. What's the difference between `//` and `/`?
2. Why does `bool(0)` return `False` but `bool("0")` return `True`?
3. What does an `elif` chain do that a series of separate `if` statements doesn't?

## Practice

1. Write a function-free script that asks for a user's age and prints whether they can vote (18+), drive (16+), or neither.
2. Print the first 20 Fibonacci numbers using a `while` loop.
3. Given a string, print it reversed without using slicing (`[::-1]`) — use a loop instead.

## Challenge

Write a simple number-guessing game: pick a random number between 1 and 100 (`random.randint`), let the user guess in a loop, and print "higher" or "lower" until they get it, counting the number of guesses.

## Where Next?

**Next: Chapter 2 covers Python's core data structures — lists, tuples, dictionaries, and sets — the containers you'll use to hold real data.**
