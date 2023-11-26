import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import json

# Load your JSON data directly
with open("Datasets/part_1.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert to pandas DataFrame
df = pd.DataFrame(data)

# Split the dataset into training and validation sets
train_data, val_data = train_test_split(df, test_size=0.5, random_state=42)

# Fine-tuning parameters
max_length = 128
batch_size = 128
epochs = 5
num_models = 5  # Number of models to create

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-turkish-uncased")

# Tokenize and format the data
encodings = tokenizer(list(df["text"]), truncation=True,
                      padding=True, max_length=max_length, return_tensors="tf")
labels = np.array(df["star"])

# Prepare TensorFlow dataset
dataset = tf.data.Dataset.from_tensor_slices((dict(encodings), labels))

# Create and fine-tune multiple models
for i in range(num_models):
    # Resample the data for each iteration
    subset_size = len(df) // 2
    resampled_indices = np.random.choice(len(df), subset_size, replace=True)
    resampled_data = df.iloc[resampled_indices]

    # Convert star ratings to integer labels
    resampled_data['label'] = resampled_data['star'].apply(lambda x: int(x))

    # Load a new tokenizer for each iteration
    tokenizer = BertTokenizer.from_pretrained(
        "ytu-ce-cosmos/turkish-small-bert-uncased")

    # Prepare resampled TensorFlow dataset
    resampled_encodings = tokenizer(list(
        resampled_data["text"]), truncation=True, padding=True, max_length=max_length, return_tensors="tf")
    resampled_labels = np.array(resampled_data["label"])
    resampled_dataset = tf.data.Dataset.from_tensor_slices(
        (dict(resampled_encodings), resampled_labels))

    # Load pre-trained BERT model for each iteration
    model = TFBertForSequenceClassification.from_pretrained(
        "ytu-ce-cosmos/turkish-small-bert-uncased", num_labels=6)

    # Model compilation
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=["accuracy"])

    # Fine-tune the model
    model.fit(resampled_dataset.shuffle(1000).batch(
        batch_size), epochs=epochs, batch_size=batch_size)

    # Save the fine-tuned model
    model.save_pretrained(f"Small_Models/model_small_{i + 1}")
    tokenizer.save_pretrained(f"Small_Models/model_small_{i + 1}")

    # Tokenize and format the validation data
    val_encodings = tokenizer(list(
        val_data["text"]), truncation=True, padding=True, max_length=max_length, return_tensors="tf")
    val_labels = np.round(np.array(val_data["star"])).astype(int)
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (dict(val_encodings), val_labels))
    val_data["star"] = val_data["star"].astype(int)

    # Evaluate the model on the validation set
    val_predictions = model.predict(val_dataset.batch(batch_size))
    val_predictions = tf.argmax(val_predictions.logits, axis=1)
    val_accuracy = accuracy_score(val_data["star"], val_predictions)

    print(f"Model {i + 1} Validation Accuracy: {val_accuracy * 100:.2f}%")
