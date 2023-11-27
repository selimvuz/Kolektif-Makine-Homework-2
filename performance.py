import tensorflow as tf
import numpy as np
import json
import pandas as pd
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# JSON verilerini doğrudan yükleyin
with open("Datasets/part_10.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

train_data, val_data = train_test_split(df, test_size=0.1, random_state=42)

tokenizer = BertTokenizer.from_pretrained("Small_Models/model_small_1")

num_models = 5

models = []
for i in range(1, num_models + 1):
    model = TFBertForSequenceClassification.from_pretrained(
        f"Small_Models/model_small_{i}")
    models.append(model)

max_length = 128
batch_size = 64
val_encodings = tokenizer(list(
    val_data["text"]), truncation=True, padding=True, max_length=max_length, return_tensors="tf")
val_labels = np.array(val_data["star"]).astype(int)
val_dataset = tf.data.Dataset.from_tensor_slices(
    (dict(val_encodings), val_labels))

ensemble_predictions = np.zeros((len(val_labels), num_models))

for i, model in enumerate(models):
    model_predictions = model.predict(val_dataset.batch(batch_size))
    ensemble_predictions[:, i] = tf.argmax(model_predictions["logits"], axis=1)

ensemble_predictions_avg = np.mean(ensemble_predictions, axis=1)


ensemble_predictions_avg = np.round(ensemble_predictions_avg).astype(int)

ensemble_accuracy = accuracy_score(val_labels, ensemble_predictions_avg)
print(f"Ensemble Accuracy (Averaging): {ensemble_accuracy * 100:.2f}%")

# Calculate the majority vote (mode) for each sample
ensemble_predictions_mode = np.apply_along_axis(
    lambda x: np.bincount(np.round(x).astype(int)).argmax(), axis=1, arr=ensemble_predictions)

ensemble_accuracy = accuracy_score(val_labels, ensemble_predictions_mode)
print(f"Ensemble Accuracy (Max Voting): {ensemble_accuracy * 100:.2f}%")
