import tensorflow as tf
import numpy as np
import json
import pandas as pd
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load your JSON data directly
with open("Datasets/part_10.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert to pandas DataFrame
df = pd.DataFrame(data)

# Split the dataset into training and validation sets
train_data, val_data = train_test_split(df, test_size=0.1, random_state=42)

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("Small_Models/model_small_1")

# Number of models
num_models = 5

# Load the fine-tuned models
models = []
for i in range(1, num_models + 1):
    model = TFBertForSequenceClassification.from_pretrained(
        f"Small_Models/model_small_{i}")
    models.append(model)

# Load and preprocess validation data
max_length = 128
batch_size = 64
val_encodings = tokenizer(list(
    val_data["text"]), truncation=True, padding=True, max_length=max_length, return_tensors="tf")
val_labels = np.array(val_data["star"]).astype(int)
val_dataset = tf.data.Dataset.from_tensor_slices(
    (dict(val_encodings), val_labels))

# Ensemble predictions
ensemble_predictions = np.zeros((len(val_labels), num_models))

for i, model in enumerate(models):
    model_predictions = model.predict(val_dataset.batch(batch_size))
    ensemble_predictions[:, i] = tf.argmax(model_predictions["logits"], axis=1)

# Calculate the ensemble result by averaging predictions
ensemble_predictions_avg = np.mean(ensemble_predictions, axis=1)

# Convert the averaged predictions to integer labels (assuming they are rounded)
ensemble_predictions_avg = np.round(ensemble_predictions_avg).astype(int)

# Calculate the ensemble accuracy
ensemble_accuracy = accuracy_score(val_labels, ensemble_predictions_avg)
print(f"Ensemble Accuracy (Averaging): {ensemble_accuracy * 100:.2f}%")

# Calculate the majority vote (mode) for each sample
ensemble_predictions_mode = np.apply_along_axis(
    lambda x: np.bincount(np.round(x).astype(int)).argmax(), axis=1, arr=ensemble_predictions)

# Calculate the ensemble accuracy
ensemble_accuracy = accuracy_score(val_labels, ensemble_predictions_mode)
print(f"Ensemble Accuracy (Max Voting): {ensemble_accuracy * 100:.2f}%")
