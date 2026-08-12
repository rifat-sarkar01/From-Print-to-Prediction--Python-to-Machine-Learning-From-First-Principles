# Glossary & Further Reading

## Glossary

**Accuracy** — the percentage of predictions a model got exactly right. Misleading on imbalanced data (Chapter 11.10, 12.11).

**Activation function** — a non-linear function applied after a neural network layer's weighted sum (e.g. ReLU, sigmoid), without which stacked layers would collapse into one linear function (Chapter 13.1).

**Backpropagation** — the algorithm that computes how much each weight in a neural network contributed to the error, using the chain rule (Chapter 13.2).

**Baseline** — the simplest possible model, used as the minimum bar any real model must beat (Chapter 11.10).

**Batch** — a small subset of training data processed together in one step of training, rather than the whole dataset at once (Chapter 13.2).

**Bias (model)** — error from a model too simple to capture the real pattern; the formal term behind underfitting (Chapter 11.6).

**Class imbalance** — when one class vastly outnumbers another in a classification dataset, making accuracy an unreliable metric (Chapter 11.10).

**Classification** — predicting a category rather than a number (Chapter 11.2).

**Cross-validation** — repeatedly splitting data into train/test folds to get a more reliable performance estimate than a single split (Chapter 12.12).

**Data leakage** — when information from outside the training set (often the test set itself) improperly influences training, producing unrealistically good evaluation scores (Chapter 11.8).

**Dropout** — randomly disabling a fraction of neurons during training to reduce overfitting (Chapter 13.4).

**Embedding** — a dense vector representation of a discrete item (like a word), learned so that similar items end up close together in the vector space (Chapter 14.2).

**Epoch** — one full pass through the entire training dataset (Chapter 13.2).

**Feature** — an individual input column/variable a model uses to make predictions.

**Feature engineering** — creating new, more informative features from existing raw data (Chapter 8.11).

**Gradient** — the multi-input generalization of a derivative; tells you which direction to adjust each weight to reduce error (Chapter 10.7).

**Gradient descent** — the optimization algorithm that repeatedly nudges weights in the direction that reduces loss (Chapter 10.9).

**Hyperparameter** — a setting chosen before training (like `max_depth`), as opposed to a parameter the model learns (Chapter 12.13).

**Inference** — using a trained model to make a prediction on new data (as opposed to training).

**Label** — the correct answer associated with a training example (also called the target).

**Loss function** — a single number measuring how wrong a model's predictions are; the quantity training tries to minimize (Chapter 10.8).

**Overfitting** — a model that has memorized the training data's noise rather than its underlying pattern, performing well on training data and poorly on new data (Chapter 11.5).

**Precision** — of everything the model predicted positive, the fraction that actually was (Chapter 12.11).

**Recall** — of everything actually positive, the fraction the model correctly caught (Chapter 12.11).

**Regression** — predicting a continuous number rather than a category (Chapter 11.2).

**Regularization** — techniques (like Ridge/Lasso penalties or Dropout) that discourage a model from fitting noise, reducing overfitting (Chapter 12.1, 13.4).

**Target** — the value a supervised model is trying to predict (also called the label).

**Underfitting** — a model too simple to capture the real pattern in the data, performing poorly even on training data (Chapter 11.5).

**Validation set** — data used to compare models or tune settings during development, kept separate from the final test set (Chapter 11.4).

**Variance (model)** — error from a model too sensitive to the specific training data it saw; the formal term behind overfitting (Chapter 11.6).

**Vectorization** — expressing a computation as whole-array operations instead of an explicit loop, letting NumPy run it in fast compiled code (Chapter 7.5).

## References & Further Reading

This book is a starting point, not the final word. These are the sources worth going to next:

**Official documentation** (the best-maintained, most current reference for anything in this book):

- Python — docs.python.org
- NumPy — numpy.org/doc
- pandas — pandas.pydata.org/docs
- Matplotlib — matplotlib.org, Seaborn — seaborn.pydata.org
- scikit-learn — scikit-learn.org (the User Guide, not just the API reference, is excellent)
- PyTorch — pytorch.org/docs
- TensorFlow/Keras — keras.io
- Hugging Face — huggingface.co/docs

**Foundational papers**, if you want to go beneath the library calls:

- "Attention Is All You Need" (Vaswani et al., 2017) — the paper that introduced the Transformer architecture behind Chapter 13.8.
- "Deep Residual Learning for Image Recognition" (He et al., 2015) — introduced ResNets, foundational to modern CNNs.
- "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2014) — the optimizer used as the default throughout Chapter 13.
- "Random Forests" (Breiman, 2001) — the original paper behind Chapter 12.4.

**Where to practice:**

- Kaggle (kaggle.com) — free datasets and competitions spanning every topic in this book, with public notebooks to learn from.
- Papers With Code (paperswithcode.com) — recent research paired with runnable implementations.

## Where to Go From Here

- **Practice on real datasets.** Kaggle hosts free datasets and competitions spanning every topic in this book.
- **Read library documentation directly.** The docs above are exceptionally well-written and go deeper than any single book can.
- **Build end-to-end projects.** The fastest way to internalize this material is picking a real problem — from a messy raw dataset through a deployed prediction endpoint — and living through every chapter of this book in the process, including the parts that don't work on the first try.

---

*This concludes the book. You've gone from `print("Hello, world!")` to fine-tuning transformer models and serving predictions over an API — everything in between is now yours to build with.*
