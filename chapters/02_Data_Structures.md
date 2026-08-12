# Chapter 2: Data Structures

Python has four built-in **collection types** — containers that hold multiple values. Choosing the right one is one of the most important skills in writing good Python (and fast ML code later).

| Type | Ordered? | Mutable? | Duplicates? | Syntax |
|------|----------|----------|-------------|--------|
| `list` | Yes | Yes | Yes | `[1, 2, 3]` |
| `tuple` | Yes | No | Yes | `(1, 2, 3)` |
| `dict` | Yes (3.7+) | Yes | Keys: No | `{"a": 1}` |
| `set` | No | Yes | No | `{1, 2, 3}` |

## 2.1 Lists

The workhorse of Python — an ordered, changeable collection.

```python
fruits = ["apple", "banana", "cherry"]

fruits.append("date")        # adds to the end → ["apple","banana","cherry","date"]
fruits.insert(1, "avocado")  # inserts at index 1
fruits.remove("banana")      # removes the first matching value
fruits.pop()                 # removes & returns the LAST item
fruits.pop(0)                # removes & returns the item at index 0
fruits.sort()                # sorts in place, alphabetically/numerically
fruits.reverse()             # reverses in place
len(fruits)                  # number of items
"apple" in fruits            # True/False — membership test
fruits[1:3]                  # slicing, same rules as strings
```

Lists can hold **mixed types** and even other lists (nested lists):
```python
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
matrix[1][2]   # 6 — row 1, column 2
```

### List Comprehensions

A compact way to build a list from another iterable — arguably the most "Pythonic" piece of syntax:

```python
squares = [x**2 for x in range(10)]
# equivalent to:
squares = []
for x in range(10):
    squares.append(x**2)

evens = [x for x in range(20) if x % 2 == 0]   # with a filter condition
pairs = [(x, y) for x in range(3) for y in range(2)]  # nested loops
```
Read it as: `[EXPRESSION for ITEM in ITERABLE if CONDITION]`.

## 2.2 Tuples

Like a list, but **immutable** — once created, it can't be changed.

```python
point = (3, 4)
x, y = point          # "unpacking" — x=3, y=4
point[0]               # 3
# point[0] = 5         # ERROR — tuples can't be modified
```

**Why use a tuple instead of a list?** Three reasons: (1) it signals to other programmers "this data shouldn't change," (2) it's slightly faster and uses less memory, (3) tuples can be used as dictionary keys and set elements — lists cannot, because those require immutable elements.

## 2.3 Dictionaries

Key-value pairs — Python's version of a hash map. Extremely important; most real-world and ML data gets passed around as dictionaries (or things built on them).

```python
person = {
    "name": "Alice",
    "age": 30,
    "city": "Boston"
}

person["name"]              # "Alice" — access by key
person["email"] = "a@x.com" # adds a new key-value pair
person["age"] = 31          # updates an existing key
del person["city"]          # removes a key
person.get("phone", "N/A")  # "N/A" — safe access, no error if key is missing
person.keys()               # dict_keys(['name', 'age', 'email'])
person.values()             # dict_values(['Alice', 31, 'a@x.com'])
person.items()              # dict_items([('name','Alice'), ('age',31), ...])

for key, value in person.items():
    print(f"{key}: {value}")
```

### Dictionary Comprehensions
```python
squares = {x: x**2 for x in range(5)}   # {0:0, 1:1, 2:4, 3:9, 4:16}
```

## 2.4 Sets

An unordered collection of **unique** values — automatically drops duplicates. Optimized for fast membership testing and mathematical set operations.

```python
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

a | b   # union: {1,2,3,4,5,6}
a & b   # intersection: {3,4}
a - b   # difference: {1,2}
a ^ b   # symmetric difference: {1,2,5,6}

nums = [1, 2, 2, 3, 3, 3]
unique = set(nums)   # {1, 2, 3} — instant deduplication
```

## 2.5 Choosing the Right Structure

- Need order and might change the contents? → **list**
- Need order but the contents must never change (e.g. coordinates, RGB values)? → **tuple**
- Need fast lookup by a label/key? → **dict**
- Need uniqueness or fast membership checks, don't care about order? → **set**

## 2.6 Unpacking & the `*` / `**` Operators

```python
a, b, c = [1, 2, 3]              # basic unpacking
first, *rest = [1, 2, 3, 4]      # first=1, rest=[2,3,4]

def add(a, b):
    return a + b

nums = [3, 5]
add(*nums)          # unpacks list into positional args → add(3, 5)

info = {"a": 3, "b": 5}
add(**info)         # unpacks dict into keyword args → add(a=3, b=5)
```

---

## What You Learned

- Lists, tuples, dictionaries, and sets — what each is optimized for, and how to choose between them
- List and dictionary comprehensions
- Unpacking, and the `*`/`**` operators for gathering and spreading arguments

## Code to Remember

```python
[x for x in items if condition]          # list comprehension
{k: v for k, v in pairs}                 # dict comprehension
a, *rest = items                         # unpacking with a "gather" variable
```

## Common Mistakes

- Using a list when a set would do — checking `x in my_list` is slow on large lists (it checks every element); `x in my_set` is nearly instant.
- Trying to modify a tuple (`point[0] = 5`) and being surprised by the `TypeError` — that's the entire point of a tuple.
- Using a mutable default argument like `def f(items=[])` — the same list gets reused across every call, a classic Python trap covered more in Chapter 3.

## Quick Check

1. Why can't a list be used as a dictionary key, but a tuple can?
2. What does `{1, 2, 2, 3}` evaluate to, and why?
3. When would you reach for a dictionary comprehension instead of a `for` loop?

## Practice

1. Given a list of words, build a dictionary mapping each word to its length using a dictionary comprehension.
2. Write a one-line list comprehension that returns all numbers from 1-100 divisible by both 3 and 5.
3. Given two lists of equal length, zip them into a dictionary (hint: look up `zip()` and `dict()`).

## Challenge

Given a list of dictionaries representing people (`{"name": ..., "age": ...}`), write a comprehension that returns just the names of everyone over 30, sorted alphabetically.

## Where Next?

**Next: Chapter 3 covers writing your own functions and organizing code into modules.**
