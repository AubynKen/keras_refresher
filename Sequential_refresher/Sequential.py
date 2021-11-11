# %%
import numpy as np
from random import randint
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import losses
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers.core import Dense

# %%
train_samples = []
train_labels = []

for _ in range(50):
    # 5 percent of younger individuals experience side effects
    younger_age = randint(13, 64)
    train_samples.append(younger_age)
    train_labels.append(1)

    # 5 % of older individuals experience no side effects
    older_age = randint(65, 100)
    train_samples.append(older_age)
    train_labels.append(0)

for _ in range(950):
    # 95 % of younger individuals experience no side effect
    younger_age = randint(13, 64)
    train_samples.append(younger_age)
    train_labels.append(0)

    # 95% of older individuals experience side effects
    older_age = randint(65, 100)
    train_samples.append(older_age)
    train_labels.append(1)

# %%

train_samples = np.array(train_samples)
train_labels = np.array(train_labels)
train_samples, train_labels = shuffle(train_samples, train_labels)
# %%
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1, 1))
# %%
model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
])

# %%
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# %%

model.fit(x=train_samples, y=train_labels, validation_split=0.1, batch_size=10, epochs=30, verbose=2)
# %%

test_samples = []
test_labels = []

for _ in range(10):
    # 5 percent of younger individuals experience side effects
    younger_age = randint(13, 64)
    test_samples.append(younger_age)
    test_labels.append(1)

    # 5 % of older individuals experience no side effects
    older_age = randint(65, 100)
    test_samples.append(older_age)
    test_labels.append(0)

for _ in range(190):
    # 95 % of younger individuals experience no side effect
    younger_age = randint(13, 64)
    test_samples.append(younger_age)
    test_labels.append(0)

    # 95% of older individuals experience side effects
    older_age = randint(65, 100)
    test_samples.append(older_age)
    test_labels.append(1)


# %%
test_samples, test_labels = shuffle(test_samples, test_labels)
# %%
predictions = model.predict(x=test_samples, batch_size=10)
# %%
rounded_predictions = np.argmax(predictions, axis=-1)
# %%
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import itertools

# %%
cm = confusion_matrix(test_labels, rounded_predictions)
# %%
disp = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"])
# %%
disp.plot()
plt.show()
# %%
model_path = "./models/medical_trial_model.h5"
import os.path
if not os.path.isfile(model_path):
    model.save(model_path)
# %%
from tensorflow.keras.models import load_model
model = load_model(model_path)
# %%
