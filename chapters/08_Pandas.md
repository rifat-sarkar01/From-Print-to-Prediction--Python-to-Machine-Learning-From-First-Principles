# Chapter 8: Pandas — Data Analysis in Python

```bash
pip install pandas
```
```python
import pandas as pd     # universal convention
```

Pandas gives you two core structures: the **Series** (a single labeled column of data) and the **DataFrame** (a full table — rows and labeled columns, like a spreadsheet or SQL table). It's built directly on NumPy, so everything from Chapter 7 about vectorized operations still applies underneath.

## 8.1 Series

```python
s = pd.Series([10, 20, 30], index=["a", "b", "c"])
s["a"]        # 10 — access by label
s.values      # array([10, 20, 30]) — the underlying NumPy array
```

## 8.2 Creating DataFrames

```python
data = {
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
    "city": ["NYC", "LA", "Chicago"]
}
df = pd.DataFrame(data)

print(df)
#       name  age     city
# 0    Alice   25      NYC
# 1      Bob   30       LA
# 2  Charlie   35  Chicago
```

## 8.3 Loading Data

```python
df = pd.read_csv("data.csv")
df = pd.read_excel("data.xlsx")
df = pd.read_json("data.json")
df = pd.read_sql("SELECT * FROM users", connection)

df.to_csv("output.csv", index=False)     # save back out
```

## 8.4 Exploring a DataFrame

```python
df.head()          # first 5 rows
df.tail(10)         # last 10 rows
df.shape            # (rows, columns)
df.info()           # column names, types, non-null counts — always run this first
df.describe()       # count, mean, std, min, quartiles, max for numeric columns
df.columns          # list of column names
df.dtypes           # data type of each column
df.isnull().sum()   # count of missing values PER COLUMN — always check this early
```

## 8.5 Selecting Data

```python
df["age"]                     # a single column (returns a Series)
df[["name", "age"]]           # multiple columns (returns a DataFrame)
df.iloc[0]                    # row by INTEGER position
df.iloc[0:3]                  # rows 0-2 by position
df.loc[0]                     # row by LABEL (index value — often the same as position, but not always)
df.loc[0, "name"]             # specific cell, by label
df[df["age"] > 28]            # boolean filtering — rows where age > 28
df[(df["age"] > 25) & (df["city"] == "NYC")]   # combine conditions with & / | (NOT and/or)
```
`.loc` uses labels, `.iloc` uses integer positions — the single most common source of confusion for Pandas beginners. When the index is the default 0,1,2..., they often look identical, which is exactly what makes the difference easy to miss until it isn't.

## 8.6 Modifying Data

```python
df["age_next_year"] = df["age"] + 1              # add a new column
df["is_adult"] = df["age"] >= 18                  # a boolean column
df.drop("city", axis=1, inplace=True)             # remove a column (axis=1); inplace modifies df directly
df.drop(0, axis=0)                                 # remove a row (axis=0)
df.rename(columns={"name": "full_name"}, inplace=True)
df["age"] = df["age"].astype(float)                # change a column's data type
```

## 8.7 Handling Missing Data

```python
df.dropna()                          # drop rows with ANY missing value
df.dropna(subset=["age"])            # drop rows missing specifically in "age"
df.fillna(0)                         # replace missing values with 0
df["age"].fillna(df["age"].mean(), inplace=True)   # fill with the column's mean — very common
df.duplicated().sum()                # count of duplicate rows
df.drop_duplicates(inplace=True)
```

## 8.8 Grouping and Aggregating

```python
df.groupby("city")["age"].mean()          # average age PER city
df.groupby("city").agg({"age": "mean", "name": "count"})
df.groupby("city").size()                  # row count per group

df["age"].value_counts()                   # frequency of each unique value
```
`groupby` follows the **split-apply-combine** pattern: split the data into groups by some key, apply a function to each group independently, then combine the results back together.

## 8.9 Merging & Joining

```python
pd.concat([df1, df2])                      # stack DataFrames on top of each other
pd.merge(df1, df2, on="id", how="inner")   # SQL-style join
# how: "inner" (only matching rows), "left", "right", "outer" (all rows, fill gaps with NaN)
```

## 8.10 Applying Functions

```python
df["age_group"] = df["age"].apply(lambda x: "adult" if x >= 18 else "minor")

def categorize(row):
    if row["age"] > 30 and row["city"] == "NYC":
        return "target"
    return "other"

df["category"] = df.apply(categorize, axis=1)   # axis=1 → apply row by row
```
**Performance note:** `.apply()` with a Python function is convenient but slow on large data because it can't use NumPy's compiled vectorization. Prefer direct vectorized operations (`df["age"] * 2`) when possible; reach for `.apply()` mainly when the logic genuinely can't be vectorized.

## 8.11 A Worked Example: Cleaning a Real Dataset

```python
df = pd.read_csv("titanic.csv")

df.info()                                       # check types & missing values first
df["Age"].fillna(df["Age"].median(), inplace=True)   # fill missing ages
df.drop(columns=["Cabin"], inplace=True)         # drop a column that's mostly empty
df.drop_duplicates(inplace=True)

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1     # feature engineering
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)
# turns categorical text columns into 0/1 numeric columns models can actually use
```

---

## What You Learned

- `Series` and `DataFrame`, and loading data from CSV/Excel/JSON/SQL
- Selecting data with `.loc`/`.iloc`, and boolean filtering
- Handling missing data (`dropna`, `fillna`) and duplicates
- `groupby` (split-apply-combine), merging/joining, and `.apply()`
- Cleaning a real, messy dataset end to end

## Code to Remember

```python
df.info()                                   # always run this first on new data
df.groupby("col")["target"].mean()          # split-apply-combine
df["age"].fillna(df["age"].median(), inplace=True)
pd.get_dummies(df, columns=["cat_col"], drop_first=True)
```

## Common Mistakes

- Confusing `.loc` (label-based) and `.iloc` (position-based) — the single most common Pandas bug for beginners.
- Filling missing values with `0` or the mean *before* checking why they're missing — sometimes "missing" itself is meaningful information (e.g. a customer who never made a purchase), and filling it in erases that signal.
- Using `.apply()` with a Python function when a vectorized operation would work — same performance trap as Chapter 7, now with a DataFrame instead of an array.

## Quick Check

1. What's the difference between `df.loc[0]` and `df.iloc[0]` when the index isn't the default 0,1,2...?
2. Why is checking `df.isnull().sum()` usually one of the first things you do with new data?
3. What does `pd.get_dummies` do, and why does it matter for categorical features?

## Practice

1. Load any CSV (or reuse the Titanic example) and print the top 3 most common values in a categorical column with `.value_counts()`.
2. Group a dataset by one categorical column and compute the mean and count of a numeric column per group.
3. Find and remove duplicate rows in a DataFrame, then verify the row count dropped.

## Challenge

Take a dataset with at least one missing-data column and one categorical column, and build a complete cleaning pipeline: fill or drop missing values with a documented reason for each choice, encode categoricals, and engineer at least one new feature from existing columns.

## Where Next?

**Next: Chapter 9 covers turning this data into charts with Matplotlib and Seaborn — essential for understanding your data before modeling it.**
