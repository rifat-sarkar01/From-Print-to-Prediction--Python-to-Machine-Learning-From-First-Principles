# Chapter 14: Advanced ML — NLP, Transformers & Deployment

## 14.1 Classic NLP Preprocessing

Before feeding text to a model, it typically needs to become numbers.

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

texts = ["I love this movie", "This movie was terrible", "Great acting and plot"]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)     # sparse matrix: rows=documents, columns=words
vectorizer.get_feature_names_out()      # the vocabulary it learned
```
- **Bag of words / `CountVectorizer`**: represents each document as word counts, ignoring order.
- **TF-IDF** (`TfidfVectorizer`, Term Frequency–Inverse Document Frequency): like word counts, but down-weights words that appear in *many* documents (like "the," "and") and up-weights words that are distinctive to a specific document — generally a better default than raw counts.

```python
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X, labels)     # classic, fast text classifier — spam detection, sentiment, topic tagging
```

**Tokenization, stemming, and lemmatization** (using `nltk` or `spaCy`):
```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats are running quickly")

[token.text for token in doc]              # tokenization: ['The','cats','are','running','quickly']
[token.lemma_ for token in doc]             # lemmatization: ['the','cat','be','run','quickly']
[(token.text, token.pos_) for token in doc] # part-of-speech tagging
[(ent.text, ent.label_) for ent in doc.ents]  # named entity recognition
```

## 14.2 Word Embeddings

Instead of treating words as arbitrary IDs, embeddings represent each word as a dense vector of numbers, learned so that words with similar meanings end up close together in that vector space (famously: `king - man + woman ≈ queen`).

```python
model = keras.Sequential([
    layers.Embedding(input_dim=vocab_size, output_dim=128),   # learns embeddings during training
    layers.GlobalAveragePooling1D(),
    layers.Dense(1, activation="sigmoid")
])
```
Modern practice almost always uses **pretrained** embeddings/models rather than learning from scratch (see 14.3) — plain text data is rarely enough to learn good general-purpose word meanings on its own.

## 14.3 Using Pretrained Transformers with Hugging Face

```bash
pip install transformers
```
The `transformers` library gives you one-line access to thousands of pretrained models for classification, generation, translation, summarization, and more.

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I absolutely loved this product!")
# [{'label': 'POSITIVE', 'score': 0.9998}]

summarizer = pipeline("summarization")
summarizer(long_article_text, max_length=100, min_length=30)

generator = pipeline("text-generation", model="gpt2")
generator("Once upon a time,", max_length=50)
```
`pipeline(...)` downloads a pretrained model and wraps preprocessing + inference + postprocessing into one call — the fastest way to get a strong baseline for most NLP tasks without training anything yourself.

### Fine-Tuning a Pretrained Model

For a task-specific dataset, you typically start from a pretrained model like **BERT** and continue training it on your own labeled data (far less data and compute needed than training from scratch):

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

encodings = tokenizer(list(texts), truncation=True, padding=True, return_tensors="pt")

training_args = TrainingArguments(
    output_dir="./results", num_train_epochs=3, per_device_train_batch_size=16
)
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
trainer.train()
```

## Mini Project: Sentiment Classifier

Build this two ways, back to back — it's the fastest way to feel the difference between "classic" and "pretrained" NLP in your own hands.

**Steps:**

1. Find a small labeled text dataset (movie or product reviews work well — several are one line away via `sklearn.datasets` or Hugging Face's `datasets` library).
2. **Classic approach:** `TfidfVectorizer` (14.1) → `LogisticRegression` or `MultinomialNB` (12.7). Evaluate with `classification_report` (12.11).
3. **Pretrained approach:** run the same test set through `pipeline("sentiment-analysis")` (14.3) with zero training.
4. Compare accuracy, and think about the trade-offs beyond the scoreboard: training time, the need for labeled data at all, and inference speed.
5. **Stretch:** fine-tune a small pretrained model on your specific dataset and see whether it beats the zero-shot pipeline.

**Common mistake:** comparing the two approaches on accuracy alone. The classic model trains in seconds on a laptop CPU; the pretrained model needed no labeled data whatsoever to get a reasonable answer. "Which is better" depends entirely on what you have — lots of labeled data and speed requirements, or almost none of either.

## 14.4 The Deployment Pipeline

A model that only exists in a notebook helps nobody. Here's the path from a trained model to something serving real predictions:

```
Train  →  Save  →  Load  →  API  →  Test  →  Deploy  →  Monitor
```

The next three sections walk through **Save → Load → API** concretely; 14.7 covers **Deploy → Monitor**.

### Save: Persisting Preprocessing Alongside the Model

A trained model is useless without the *exact* preprocessing that produced its training data — this is a common real-world deployment bug (mismatched preprocessing between training and serving).

```python
import joblib

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")           # save the fitted scaler too, not just the model
```

### Load: Reconstructing the Full Pipeline

```python
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
new_data_scaled = scaler.transform(new_data)   # use the SAME fitted scaler, never a fresh one
prediction = model.predict(new_data_scaled)
```

## 14.5 API: Serving a Model Over HTTP

The most common way to make a trained model usable by other applications:

```python
# pip install flask
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = scaler.transform([data["features"]])
    prediction = model.predict(features)
    return jsonify({"prediction": prediction.tolist()})

if __name__ == "__main__":
    app.run(port=5000)
```
A client anywhere can now `POST` data to `http://yourserver:5000/predict` and get a prediction back as JSON — this is the fundamental pattern behind how most production ML systems expose models to the rest of an application.

### Test

Before anything gets called "deployed," hit the endpoint the way a real client would — a quick sanity check that catches an enormous fraction of deployment bugs:
```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [1500, 3, 2]}'
```

## 14.6 Deploy: Packaging with Docker

**Intuition:** "it works on my machine" is the single most common deployment failure — a **container** packages your code together with its exact dependencies and environment, so it runs identically anywhere.

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```
```bash
docker build -t my-model-api .
docker run -p 5000:5000 my-model-api
```
This isn't a DevOps book, so that's the whole picture you need: a Dockerfile describes the environment, `docker build` packages it, `docker run` starts it — anywhere Docker runs, your API runs identically.

## 14.7 Monitor: MLOps Concepts Worth Knowing

Deployment isn't the finish line — a model quietly degrading in production is far more common than one that fails loudly.

- **Experiment tracking** (e.g. MLflow, Weights & Biases): logging every training run's hyperparameters, metrics, and artifacts so results are reproducible and comparable.
- **Model versioning**: treating trained models like code — tracked, tagged, and rollback-able, since a "better" model on paper can still perform worse in the real world.
- **Data drift**: real-world data gradually shifts away from what a model was trained on (user behavior changes, seasons change, a pandemic happens) — production models need ongoing monitoring, not a one-time deployment.
- **A/B testing**: routing a small percentage of real traffic to a new model version and comparing outcomes against the current one before fully switching over.

**Real world:** a credit risk model trained on pre-2020 data and never re-evaluated is a textbook drift failure — the economic conditions the model learned from no longer describe the world it's making decisions in. Monitoring isn't paranoia; it's the same overfitting-to-a-stale-snapshot problem from Chapter 11.5, playing out over months instead of epochs.

## Final Project: End-to-End ML Application

This is the capstone — string together nearly everything in this book into one real, working system.

**Goal:** pick a problem (a Customer Churn Predictor and a House Price Predictor are both good choices — plenty of clean datasets exist for both), and take it the entire distance:

1. **Data**: load, clean, and explore it (Chapters 8-9).
2. **Baseline**: fit a `Dummy` model first (Chapter 11.10).
3. **Model**: train and compare at least two real algorithms with cross-validation (Chapter 12).
4. **Tune**: run `GridSearchCV` or `RandomizedSearchCV` on your best candidate (Chapter 12.13).
5. **Evaluate**: report the metrics that actually matter for your problem, not just accuracy (Chapter 12.11).
6. **Save**: persist the final model and its preprocessing together (14.4).
7. **Serve**: wrap it in a Flask API with a `/predict` endpoint (14.5), and test it with `curl`.
8. **Document**: a short README stating the problem, the baseline, the final metric, and how to run the API — treat this as a real deliverable, not an afterthought.

**Where next after this:** swap in a different dataset and do it again. The second time through this list takes a fraction of the time the first one did — that's the actual sign you've learned the workflow, not just the syntax.

---

## What You Learned

- Classic NLP preprocessing (bag of words, TF-IDF) versus pretrained transformer pipelines
- Word embeddings, and why pretrained embeddings usually beat learning your own from scratch
- The full deployment pipeline: save, load, serve over an API, test, containerize, and monitor
- Core MLOps vocabulary: experiment tracking, versioning, data drift, A/B testing
- Built a sentiment classifier two ways, and a complete end-to-end application

## Common Mistakes

- Deploying a model without saving its exact preprocessing pipeline alongside it
- Treating deployment as the finish line instead of the start of monitoring
- Reaching for a large pretrained model when a `TfidfVectorizer` + `LogisticRegression` baseline would answer the question in a tenth of the time

## Quick Check

1. Why does a saved model need its fitted scaler saved alongside it, not just its weights?
2. What problem does Docker solve that a `requirements.txt` file alone doesn't?
3. Why is data drift a genuine risk even for a model that scored well at launch?

## Practice

1. Save and reload a trained Scikit-learn model with `joblib`, and confirm its predictions match before and after.
2. Build a one-endpoint Flask API for any model you've trained in this book, and test it with `curl`.
3. Write a Dockerfile for that API and run it locally.

## Challenge

Take the Titanic classifier from Chapter 12's mini project and turn it into a served API: save the pipeline, load it in a Flask app, and write a `curl` command that submits a new passenger's details and gets a survival prediction back.

## Where Next?

**Next:** the Glossary and Further Reading in Chapter 15 close out the book — quick-reference definitions and where to go deeper on anything covered here.
