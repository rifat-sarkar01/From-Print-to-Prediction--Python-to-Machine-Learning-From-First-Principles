# Chapter 6: File I/O & the Standard Library

## 6.1 Reading and Writing Files

```python
# Writing
with open("notes.txt", "w") as f:      # "w" = write (overwrites existing content)
    f.write("Hello\n")
    f.write("World\n")

# Appending
with open("notes.txt", "a") as f:      # "a" = append (adds to the end)
    f.write("More text\n")

# Reading
with open("notes.txt", "r") as f:      # "r" = read (default mode)
    contents = f.read()                # whole file as one string
    
with open("notes.txt", "r") as f:
    for line in f:                     # memory-efficient: reads one line at a time
        print(line.strip())

with open("notes.txt", "r") as f:
    lines = f.readlines()              # list of every line, including "\n" characters
```

## 6.2 Working with CSV Files

```python
import csv

with open("data.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)         # each row is a list of strings

with open("data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["name", "age"])
    writer.writerow(["Alice", 30])

# DictReader is often more convenient — rows come back as dictionaries
with open("data.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row["name"])
```
In practice, you'll almost always use **Pandas** (Chapter 8) for CSVs in ML work — `csv` is worth knowing but Pandas handles messy real-world data far better.

## 6.3 JSON

JSON (JavaScript Object Notation) is the standard format for configs, API responses, and many ML model metadata files.

```python
import json

data = {"name": "Alice", "age": 30, "hobbies": ["reading", "coding"]}

json_string = json.dumps(data, indent=2)    # Python object → JSON string
print(json_string)

with open("data.json", "w") as f:
    json.dump(data, f, indent=2)            # write directly to a file

with open("data.json", "r") as f:
    loaded = json.load(f)                   # JSON file → Python dict
```

## 6.4 `os` and `pathlib` — Working with the File System

```python
import os

os.getcwd()                    # current working directory
os.listdir(".")                # list files in a directory
os.path.exists("data.csv")     # True/False
os.makedirs("output", exist_ok=True)   # create a folder (no error if it already exists)
os.environ.get("API_KEY")      # read an environment variable

from pathlib import Path       # the modern, object-oriented alternative to os.path
p = Path("data") / "raw" / "file.csv"   # builds a path with proper separators
p.exists()
p.parent                       # the containing folder
p.suffix                       # ".csv"
```

## 6.5 `datetime`

```python
from datetime import datetime, timedelta

now = datetime.now()
print(now)                                  # 2026-08-11 14:30:00.123456
print(now.strftime("%Y-%m-%d"))             # "2026-08-11" — format as string
parsed = datetime.strptime("2026-08-11", "%Y-%m-%d")   # parse a string into a date

tomorrow = now + timedelta(days=1)          # date arithmetic
```

## 6.6 Regular Expressions (`re`)

Pattern matching in text — used heavily in data cleaning and NLP preprocessing.

```python
import re

text = "Contact us at support@example.com or sales@example.com"

emails = re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
# ['support@example.com', 'sales@example.com']

re.sub(r"\d+", "#", "I have 3 apples and 12 oranges")
# "I have # apples and # oranges"

match = re.search(r"(\d+)-(\d+)", "Call 555-1234")
if match:
    print(match.group())    # "555-1234"
    print(match.group(1))   # "555"
```
Quick reference: `\d` = digit, `\w` = word character, `\s` = whitespace, `+` = one or more, `*` = zero or more, `?` = optional, `.` = any character, `^`/`$` = start/end of string.

## 6.7 `itertools` and `functools`

```python
from itertools import combinations, permutations, chain, product

list(combinations([1, 2, 3], 2))    # [(1,2), (1,3), (2,3)] — order doesn't matter
list(permutations([1, 2, 3], 2))    # [(1,2),(1,3),(2,1),(2,3),(3,1),(3,2)] — order matters
list(chain([1, 2], [3, 4]))         # [1, 2, 3, 4] — flattens multiple iterables
list(product([1, 2], ["a", "b"]))   # [(1,'a'),(1,'b'),(2,'a'),(2,'b')] — cartesian product

from functools import lru_cache

@lru_cache(maxsize=None)     # caches results — huge speedup for expensive repeated calls
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

## 6.8 `collections`

```python
from collections import Counter, defaultdict, namedtuple

Counter(["a", "b", "a", "c", "a"])   # Counter({'a': 3, 'b': 1, 'c': 1})

dd = defaultdict(list)               # auto-creates a default value for missing keys
dd["fruits"].append("apple")         # no KeyError, even though "fruits" didn't exist yet

Point = namedtuple("Point", ["x", "y"])
p = Point(3, 4)
print(p.x, p.y)     # 3 4 — tuple with named fields, lightweight alternative to a class
```

## Mini Project: Expense Tracker

**Goal:** a CLI tool that lets you log expenses, saves them to a JSON file so they persist between runs, and prints a summary.

**Steps:**

1. Design a record shape: `{"date": "2026-08-12", "category": "food", "amount": 12.50}`.
2. Write `load_expenses(path)` and `save_expenses(path, expenses)` using `json.load`/`json.dump` — handle the case where the file doesn't exist yet (first run).
3. Build a small menu loop (reusing the pattern from Chapter 3's calculator): add an expense, list all expenses, show a total by category (a `Counter` or `defaultdict` from `collections` makes this a few lines).
4. Use `datetime.now()` to stamp each entry automatically instead of asking the user to type a date.
5. **Stretch:** add a monthly summary using `strftime("%Y-%m")` to group entries by month.

**Common mistake:** overwriting the whole file with just the newest entry instead of loading the existing list, appending, and saving the full list back — a very easy bug to introduce with file-based persistence, and one worth deliberately triggering once so you recognize it later.

---

## What You Learned

- Reading and writing text files, CSV, and JSON
- `os`/`pathlib` for navigating the file system
- `datetime` for working with dates and times
- Regular expressions (`re`) for pattern matching in text
- `itertools`, `functools.lru_cache`, and `collections` (`Counter`, `defaultdict`, `namedtuple`)

## Code to Remember

```python
with open(path) as f:          # always use `with` for files — guarantees closing
    data = json.load(f)

Counter(items)                 # instant frequency count
defaultdict(list)              # dict with an automatic default value
```

## Common Mistakes

- Opening a file without `with`, and forgetting to close it — usually harmless in a short script, but a real resource leak in anything long-running.
- Forgetting `newline=""` when writing CSVs on Windows, resulting in blank lines between rows.
- Reaching for `re` when a plain `.split()` or `.replace()` would do — regex is powerful but harder to read; use it when you actually need pattern matching.

## Quick Check

1. Why is `with open(...) as f:` preferred over calling `open()`/`close()` manually?
2. What's the difference between `json.dump` and `json.dumps`?
3. When would `defaultdict(list)` save you code compared to a plain `dict`?

## Practice

1. Write a script that reads a text file and prints a count of how many times each word appears, using `Counter`.
2. Write a function that takes a list of dictionaries and writes them to a CSV file with `csv.DictWriter`.
3. Use `re.findall` to extract all the phone numbers from a block of text.

## Challenge

Extend the Expense Tracker's category summary into a simple bar chart in the terminal — print each category name followed by a number of `█` characters proportional to its total (no external libraries needed).

## Where Next?

**Next: Chapter 7 begins the machine learning track with NumPy — the array library everything else in the Python ML ecosystem is built on top of.**
