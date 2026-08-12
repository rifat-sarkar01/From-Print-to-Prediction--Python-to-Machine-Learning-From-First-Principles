# Chapter 13: Deep Learning

## 13.1 What a Neural Network Actually Is

A neural network is layers of simple units (**neurons**) stacked together. Each neuron does something almost embarrassingly simple:

```
output = activation_function(weighted_sum_of_inputs + bias)
```

Stack enough of these simple units in enough layers, and the *network as a whole* can approximate extremely complex functions — this is the **universal approximation theorem**: a large-enough neural network can, in principle, approximate any continuous function.

- **Input layer**: your raw features (e.g. pixel values, word embeddings).
- **Hidden layers**: intermediate transformations — this is what makes it "deep" (many hidden layers).
- **Output layer**: the final prediction (a class probability, a number, etc).
- **Weights**: exactly like the `w` in linear regression (Chapter 12.1) — the numbers the network learns.
- **Activation function**: a non-linear function applied after the weighted sum. Without non-linearity, stacking layers would mathematically collapse into one big linear function — no more powerful than plain linear regression. Common choices:
  - **ReLU** (`max(0, x)`) — the default for hidden layers; fast and works well in practice.
  - **Sigmoid** (squashes to 0-1) — output layer for binary classification.
  - **Softmax** (turns a vector into probabilities that sum to 1) — output layer for multi-class classification.

## 13.2 How a Network Learns: Backpropagation

1. **Forward pass**: input flows through the network, producing a prediction.
2. **Loss calculation**: compare the prediction to the true answer using a **loss function** (e.g. mean squared error for regression, cross-entropy for classification) — a single number measuring "how wrong was this."
3. **Backward pass (backpropagation)**: using calculus (the chain rule), compute how much *each individual weight* contributed to that error.
4. **Optimizer step**: nudge every weight slightly in the direction that reduces the error (**gradient descent**). Common optimizers: `SGD`, and especially `Adam` (the default in most modern code — adapts the step size per-parameter automatically).
5. Repeat for many **epochs** (full passes through the training data), usually processing data in small **batches** rather than all at once (faster, and adds a helpful bit of noise that improves generalization).

## 13.3 Keras (via TensorFlow) — The Beginner-Friendly Path

```bash
pip install tensorflow
```
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(10,)),   # hidden layer, 64 neurons
    layers.Dense(32, activation="relu"),                       # another hidden layer
    layers.Dense(1, activation="sigmoid")                       # output layer — binary classification
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)

model.evaluate(X_test, y_test)
predictions = model.predict(X_test)
```
- `Dense` = a **fully connected layer** — every neuron connects to every neuron in the previous layer.
- `input_shape=(10,)` — only needed on the first layer, telling the network how many features to expect.
- `.compile()` configures *how* training will happen (optimizer, loss, metrics) but doesn't train yet.
- `.fit()` actually trains the model, and `history` records loss/accuracy at every epoch — plot it to check for overfitting (training accuracy climbing while validation accuracy stalls or drops is the classic warning sign).

```python
import matplotlib.pyplot as plt
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="validation")
plt.legend(); plt.show()
```

### Regression with Keras
```python
model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(10,)),
    layers.Dense(32, activation="relu"),
    layers.Dense(1)                     # NO activation on the output — raw numeric prediction
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
```

### Multi-Class Classification with Keras
```python
model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(10,)),
    layers.Dense(10, activation="softmax")   # one output neuron PER CLASS, probabilities sum to 1
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# "sparse_categorical_crossentropy" expects integer labels (0,1,2...)
# "categorical_crossentropy" expects one-hot encoded labels instead
```

## 13.4 Preventing Overfitting in Neural Networks

```python
layers.Dropout(0.3)     # randomly "turns off" 30% of neurons each training step — forces
                         # the network not to over-rely on any single neuron

keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
# stops training automatically once validation loss stops improving for 5 epochs straight
```
```python
model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(10,)),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])

early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stop])
```

## 13.5 PyTorch — The Research-Standard Alternative

```bash
pip install torch
```
PyTorch is more explicit than Keras — you write the forward pass yourself, which makes it more flexible (and is why most cutting-edge research code is written in it) at the cost of more boilerplate.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):                        # defines the forward pass explicitly
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x

model = NeuralNet()
criterion = nn.BCELoss()                          # binary cross-entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

X_train_t = torch.FloatTensor(X_train)             # PyTorch needs data as Tensors, its own array type
y_train_t = torch.FloatTensor(y_train).unsqueeze(1) # reshape to a column, matching the model's output

for epoch in range(20):
    optimizer.zero_grad()                # clear gradients from the last step
    outputs = model(X_train_t)           # forward pass
    loss = criterion(outputs, y_train_t) # compute loss
    loss.backward()                      # backpropagation — computes gradients
    optimizer.step()                     # update weights using those gradients
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```
This four-line loop — `zero_grad()` → `backward()` → `step()`, surrounding a forward pass — is the literal core of essentially all deep learning training, in every framework. Keras's `.fit()` is doing exactly this internally; PyTorch just makes you write it out.

## 13.6 Convolutional Neural Networks (CNNs) — Images

Instead of connecting every pixel to every neuron (which would be enormous and ignore spatial structure), CNNs slide small learnable filters across the image, detecting local patterns like edges, textures, and eventually whole shapes as layers stack.

```python
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(64, 64, 3)),  # 32 filters, 3x3 each
    layers.MaxPooling2D((2,2)),           # downsamples, keeping the strongest signal in each region
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),                      # collapses the 2D feature maps into a 1D vector
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
```
- **`Conv2D`**: each filter learns to detect one type of local pattern; early layers learn simple things (edges, colors), deeper layers combine those into complex concepts (eyes, wheels, faces).
- **`MaxPooling2D`**: shrinks the spatial size, keeping the strongest activation in each small region — reduces computation and adds a bit of position-invariance (a cat detected slightly off-center is still detected).
- **`(64, 64, 3)`**: height, width, and 3 color channels (RGB).

**Transfer learning** — instead of training a CNN from scratch (which needs huge datasets), reuse a model pretrained on millions of images and fine-tune it for your specific task:
```python
base_model = keras.applications.MobileNetV2(
    input_shape=(224,224,3), include_top=False, weights="imagenet"
)
base_model.trainable = False    # freeze the pretrained weights

model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation="sigmoid")   # only this new final layer gets trained
])
```

## 13.7 Recurrent Neural Networks (RNNs) & LSTMs — Sequences

For data where **order matters** — text, time series, audio — RNNs process input one step at a time, carrying a "memory" (hidden state) forward from each step to the next.

```python
model = keras.Sequential([
    layers.Embedding(input_dim=10000, output_dim=64),   # turns word IDs into dense vectors
    layers.LSTM(64),                                      # LSTM = Long Short-Term Memory
    layers.Dense(1, activation="sigmoid")
])
```
Plain RNNs struggle to remember information from many steps back (the **vanishing gradient problem**). **LSTM** and **GRU** layers add gating mechanisms specifically designed to preserve important information over longer sequences — this is why they're used instead of a plain `SimpleRNN` in almost all real applications.

## 13.8 Transformers — Why They Replaced RNNs

**Intuition:** in the sentence "the trophy didn't fit in the suitcase because **it** was too big," what does "it" refer to? You resolve this instantly by relating "it" back to "trophy," skipping over every word in between. **Self-attention** is that skill, formalized: for every word, the model directly computes how relevant every *other* word in the sequence is to it, regardless of distance, and blends their information accordingly.

This is a fundamentally different strategy from an RNN's step-by-step memory (13.7) — instead of carrying information forward one word at a time and hoping it survives, every word gets a direct line to every other word. That solves RNNs' long-range memory problem, and, crucially, it's far more parallelizable on GPUs, since there's no step-by-step dependency forcing words to be processed in order.

Stack enough of these attention layers, add positional information (so the model knows word *order*, since attention alone doesn't), and you have a **Transformer** — the architecture behind GPT, BERT, and effectively all modern large language models. Building one from scratch is beyond this book's scope, but Chapter 14 covers using pretrained transformer models directly via the Hugging Face library — the standard way to work with them in practice.

## 13.9 Saving and Loading Models

```python
# Keras
model.save("my_model.keras")
loaded = keras.models.load_model("my_model.keras")

# PyTorch
torch.save(model.state_dict(), "model_weights.pth")
model.load_state_dict(torch.load("model_weights.pth"))
model.eval()    # switches to inference mode — disables dropout, etc.
```

## Mini Project: Handwritten Digit Classifier

The "hello world" of deep learning — and a good first hands-on encounter with real overfitting and real accuracy numbers, not toy examples.

**Steps:**

1. Load MNIST directly from Keras: `(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()` — 60,000 handwritten digit images, 28x28 pixels each.
2. Normalize pixel values to [0, 1] (divide by 255), and reshape for a CNN: `(28, 28, 1)`.
3. Build a small CNN (13.6): two `Conv2D`/`MaxPooling2D` pairs, then `Flatten`, then a `Dense(10, activation="softmax")` output layer for the 10 digit classes.
4. Train with `sparse_categorical_crossentropy`, track validation accuracy, and add `EarlyStopping` (13.4).
5. Plot a few misclassified digits alongside the model's (wrong) prediction — this is usually far more informative than the accuracy number alone, and a good habit for every classification project from here on.
6. **Stretch:** compare your CNN's accuracy against a plain `Dense`-only network with no convolutions at all — the gap is a direct, hands-on demonstration of why spatial structure matters for images.

A well-tuned small CNN should comfortably clear 98% test accuracy on MNIST — if yours is far below that, check normalization and label encoding first; if it's suspiciously at 100%, check for a data leak between train and test.

---

## What You Learned

- What a neuron, layer, and activation function actually compute
- Backpropagation and gradient descent as the training engine, in both Keras and raw PyTorch
- Dropout and early stopping as the standard overfitting defenses for neural nets
- CNNs for images, RNNs/LSTMs for sequences, and why Transformers replaced RNNs for most sequence tasks
- Saving and loading trained models in both frameworks

## Common Mistakes

- Forgetting to normalize input data (e.g. raw 0-255 pixel values) — networks train far more reliably on small, centered input ranges.
- Using the wrong output activation/loss pairing (Chapter 10.8) — sigmoid + binary cross-entropy for two classes, softmax + categorical cross-entropy for more than two.
- Not plotting the training history — training and validation curves diverging is the earliest, clearest overfitting signal you'll get, well before a final test score confirms it.

## Quick Check

1. Why does stacking layers with no activation function between them provide no benefit over a single layer?
2. What problem does Dropout solve, and how does it solve it?
3. Why is a Transformer more parallelizable on a GPU than an RNN?

## Practice

1. Train a small `Dense`-only network on a tabular classification dataset and plot train vs. validation accuracy across epochs.
2. Add `Dropout(0.3)` to that network and compare the overfitting gap before and after.
3. Load a pretrained image classifier (`keras.applications.MobileNetV2`) and classify a handful of your own images with it, no training required.

## Challenge

Build the Handwritten Digit Classifier mini project above, then deliberately remove the `Conv2D`/`MaxPooling2D` layers and replace them with `Dense` layers only, keeping parameter count roughly comparable. Compare accuracy and training time — this is the clearest hands-on argument for why CNNs exist.

## Where Next?

**Next: Chapter 14 covers NLP with pretrained transformers, and how to package a trained model into something that serves real predictions.**
