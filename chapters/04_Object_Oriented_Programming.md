# Chapter 4: Object-Oriented Programming (OOP)

Nearly every ML library (Scikit-learn models, PyTorch neural networks, Pandas DataFrames) is built using classes. Understanding OOP is what lets you read library source code and build your own reusable components instead of just calling other people's.

## 4.1 Classes and Objects

A **class** is a blueprint. An **object** (or **instance**) is a specific thing built from that blueprint.

```python
class Dog:
    def __init__(self, name, breed):    # constructor — runs when you create an instance
        self.name = name                # instance attribute
        self.breed = breed

    def bark(self):                     # instance method
        return f"{self.name} says Woof!"

my_dog = Dog("Rex", "Labrador")   # creates an INSTANCE of the Dog class
print(my_dog.name)                # Rex
print(my_dog.bark())              # Rex says Woof!
```

- `__init__` is the **constructor** — a special ("dunder," short for double-underscore) method Python calls automatically when you write `Dog(...)`.
- `self` refers to *the specific instance* the method is being called on. It's always the first parameter of an instance method, and Python passes it automatically — you never write it explicitly when calling `my_dog.bark()`.
- `self.name = name` stores `name` as an **attribute** *on that instance* — every `Dog` object gets its own independent `name`.

## 4.2 Class Attributes vs. Instance Attributes

```python
class Dog:
    species = "Canis familiaris"    # CLASS attribute — shared by ALL instances

    def __init__(self, name):
        self.name = name            # INSTANCE attribute — unique per object

a = Dog("Rex")
b = Dog("Fido")
print(a.species)   # Canis familiaris
print(b.species)   # Canis familiaris — same value, shared
print(a.name)       # Rex
print(b.name)       # Fido — different, per-instance
```

## 4.3 Inheritance

Lets one class reuse and extend another's behavior:

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return f"{self.name} makes a sound"

class Cat(Animal):                   # Cat INHERITS from Animal
    def speak(self):                 # OVERRIDES the parent's method
        return f"{self.name} says Meow"

class Puppy(Animal):
    def __init__(self, name, age):
        super().__init__(name)       # calls the PARENT's __init__
        self.age = age

c = Cat("Whiskers")
print(c.speak())     # Whiskers says Meow

p = Puppy("Rex", 1)
print(p.name, p.age)  # Rex 1
```
`super()` gives you access to the parent class, most commonly used to call the parent's `__init__` so you don't have to duplicate its setup logic.

## 4.4 Polymorphism

Different classes responding to the same method call, each in their own way:

```python
animals = [Cat("Tom"), Animal("Generic")]
for animal in animals:
    print(animal.speak())   # each object uses ITS OWN version of speak()
```
This is why Scikit-learn models all support `.fit()` and `.predict()` — different algorithms underneath, same shared interface. You'll rely on this constantly in Chapters 10-13.

## 4.5 Encapsulation

Python doesn't enforce strict "private" access like Java does, but uses **naming conventions**:

```python
class BankAccount:
    def __init__(self, balance):
        self._balance = balance      # single underscore = "internal, don't touch" (convention only)
        self.__pin = "1234"          # double underscore = name-mangled, harder to access accidentally

    def deposit(self, amount):
        if amount > 0:
            self._balance += amount

    def get_balance(self):
        return self._balance

account = BankAccount(100)
account.deposit(50)
print(account.get_balance())    # 150
```

## 4.6 Magic (Dunder) Methods

Special methods that let your objects work with Python's built-in syntax (`+`, `len()`, `print()`, etc):

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):                    # controls what print()/repr() shows
        return f"Vector({self.x}, {self.y})"

    def __add__(self, other):              # defines behavior for the + operator
        return Vector(self.x + other.x, self.y + other.y)

    def __eq__(self, other):               # defines behavior for ==
        return self.x == other.x and self.y == other.y

    def __len__(self):                     # defines behavior for len()
        return 2

v1 = Vector(1, 2)
v2 = Vector(3, 4)
print(v1 + v2)        # Vector(4, 6) — calls __add__
print(v1 == v2)       # False        — calls __eq__
```
This is exactly how NumPy arrays support `array1 + array2` doing element-wise math instead of Python's default (concatenating lists) — NumPy's `ndarray` class defines its own `__add__`.

## 4.7 Abstract Base Classes

Used to define a required interface that subclasses *must* implement — common in ML framework internals (e.g. custom PyTorch layers):

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass    # subclasses MUST implement this or Python raises an error

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14159 * self.radius ** 2

c = Circle(5)
print(c.area())    # 78.53975
# shape = Shape()  # ERROR — can't instantiate an abstract class directly
```

---

## What You Learned

- Classes, `__init__`, instance vs. class attributes
- Inheritance, `super()`, and method overriding
- Polymorphism — why `.fit()`/`.predict()` work identically across every Scikit-learn model
- Magic methods (`__repr__`, `__add__`, `__eq__`, `__len__`) and abstract base classes

## Common Mistakes

- Confusing a class attribute with an instance attribute — mutable class attributes (like a list) are especially dangerous, since they're silently *shared* across every instance unless set in `__init__`.
- Forgetting to call `super().__init__()` in a subclass, silently skipping the parent's setup logic.
- Overusing inheritance where composition would be simpler — a common sign is a deep chain of subclasses for what's really just a handful of independent behaviors.

## Quick Check

1. What's the difference between a class attribute and an instance attribute?
2. Why does polymorphism let you swap `RandomForestClassifier` for `LogisticRegression` without changing your training code?
3. What does `super().__init__()` actually do?

## Practice

1. Write a `Rectangle` class with `width` and `height`, plus a `.area()` and `.perimeter()` method.
2. Create a `Square` subclass of `Rectangle` that only takes one side length.
3. Give your `Rectangle` class a `__repr__` method so `print()`-ing an instance shows something useful.

## Challenge

Build a small class hierarchy for shapes (`Shape` as an abstract base class with an abstract `.area()` method, then `Circle`, `Rectangle`, `Triangle` subclasses). Write a function that takes a list of mixed shapes and returns the total area — this is polymorphism doing real work.

## Where Next?

**Next: Chapter 5 covers advanced Python patterns — decorators, generators, context managers, and proper error handling.**
