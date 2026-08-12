# Chapter 9: Data Visualization

```bash
pip install matplotlib seaborn
```

## 9.1 Matplotlib Basics

Matplotlib is the foundational plotting library — lower-level and more verbose than Seaborn, but capable of anything.

```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]

plt.plot(x, y)                  # line plot
plt.title("Squares")
plt.xlabel("x")
plt.ylabel("x squared")
plt.show()                      # renders the figure
```

### Common Plot Types
```python
plt.plot(x, y)                       # line plot — trends over a continuous variable
plt.scatter(x, y)                    # scatter plot — relationship between two variables
plt.bar(["A", "B", "C"], [3, 7, 5])  # bar chart — comparing categories
plt.hist(data, bins=20)              # histogram — distribution of one variable
plt.boxplot(data)                    # box plot — median, quartiles, outliers
```

### Customizing a Plot
```python
plt.figure(figsize=(8, 5))                       # set the figure size (width, height in inches)
plt.plot(x, y, color="blue", linestyle="--", linewidth=2, label="squares")
plt.legend()                                      # shows the label
plt.grid(True)
plt.savefig("plot.png", dpi=300)                  # save to a file
plt.show()
```

### Subplots (Multiple Charts in One Figure)
```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))   # 1 row, 2 columns

axes[0].plot(x, y)
axes[0].set_title("Line Plot")

axes[1].scatter(x, y)
axes[1].set_title("Scatter Plot")

plt.tight_layout()    # prevents overlapping labels
plt.show()
```

## 9.2 Seaborn — Statistical Plots, Built on Matplotlib

Seaborn understands Pandas DataFrames directly and produces good-looking statistical charts with far less code.

```python
import seaborn as sns
import pandas as pd

df = pd.read_csv("data.csv")

sns.histplot(data=df, x="age", bins=20, kde=True)   # histogram + smoothed density curve
sns.boxplot(data=df, x="city", y="age")              # distribution per category
sns.scatterplot(data=df, x="age", y="income", hue="city")   # hue = color-code by a category
sns.pairplot(df)                                     # every numeric column plotted against every other
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")  # correlation matrix — extremely common in ML EDA
plt.show()
```

## 9.3 The Correlation Heatmap — Your First Real ML Diagnostic

```python
correlation_matrix = df.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlations")
plt.show()
```
Values range from -1 (perfectly inverse relationship) to +1 (perfectly direct relationship), with `annot=True` printing the actual numbers on each cell. This is usually one of the first things you run on a new dataset — it flags which features might be predictive, and which pairs of features are redundant (highly correlated with each other, which can hurt certain models like linear regression).

## 9.4 Distribution Plots for Understanding a Target Variable

```python
sns.countplot(data=df, x="target_class")    # class balance — how many of each category
sns.histplot(data=df, x="price", kde=True)  # is a numeric target skewed? normal? bimodal?
```
Checking your target variable's distribution before modeling matters: a heavily imbalanced classification target (e.g. 95% "no," 5% "yes") needs special handling (Chapter 12), and a skewed numeric target often benefits from a log transform before regression.

## Mini Project: CSV Data Analyzer

**Goal:** a small script that takes any CSV file and automatically produces a one-page visual summary — genuinely useful for the "what am I even looking at" moment every real dataset starts with.

**Steps:**

1. Load the CSV with Pandas and print `.info()` and `.describe()`.
2. For every numeric column, plot a histogram; for every categorical column (under some cardinality threshold, say 15 unique values), plot a bar chart of value counts.
3. Plot a correlation heatmap for the numeric columns.
4. Arrange everything with `plt.subplots()` into a single readable grid figure instead of dozens of separate pop-ups, and save it with `plt.savefig("summary.png")`.
5. **Stretch:** automatically flag columns that are more than 30% missing, and highlight the most-correlated feature pair.

This is close to a miniature version of what tools like `pandas-profiling`/`ydata-profiling` do automatically — building it once yourself is the fastest way to actually understand what those tools are doing for you.

---

## What You Learned

- Matplotlib basics: line, scatter, bar, histogram, box plots, subplots
- Seaborn: statistical plots that understand DataFrames directly
- The correlation heatmap as a first diagnostic on any new dataset
- Checking a target variable's distribution before modeling

## Common Mistakes

- Skipping visualization and jumping straight to modeling — outliers, skew, and broken data (like a "-1" used as a missing-value placeholder) are often invisible in `.describe()` but obvious in a histogram.
- Reading too much into a correlation heatmap — correlation captures *linear* relationships only, and says nothing about causation.
- Producing a chart with unlabeled axes or no title — fine for quick exploration, not fine the moment you show it to someone else.

## Quick Check

1. Why check a correlation heatmap before training a linear model specifically?
2. What does a strongly right-skewed histogram on a target variable suggest you might need to do before regression?
3. Why is `sns.countplot` on your target class one of the first things worth plotting for a classification problem?

## Practice

1. Plot a histogram of any numeric column in a dataset of your choice, with `kde=True`.
2. Build a correlation heatmap and identify the two most correlated features.
3. Make a bar chart comparing the mean of some numeric column across categories of another column.

## Challenge

Build the CSV Data Analyzer mini project above and run it against two different real datasets (try any small dataset from Kaggle or the built-in `seaborn.load_dataset("tips")`). Compare what each summary tells you before you've written a single line of modeling code.

## Where Next?

**Next: Chapter 10 builds the mathematical foundation — vectors, matrices, gradients, and loss functions — that every algorithm from here on is written in terms of.**
