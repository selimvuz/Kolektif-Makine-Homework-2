from transformers import BertTokenizer, TFBertModel
import numpy as np
import pandas as pd
import json

# Load your JSON data directly
dataset = pd.read_json('Datasets/part_0.json')

tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-uncased')
model = TFBertModel.from_pretrained('dbmdz/bert-base-turkish-uncased')

# Initialize lists to store feature vectors
feature_vectors = []
max_sequence_length = 512

for index, row in dataset.iterrows():
    text = row['text'][:max_sequence_length - 2]
    star_rating = row['star']

    # Tokenize and encode the text
    inputs = tokenizer(text, return_tensors="tf", truncation=True,
                       max_length=max_sequence_length, padding='max_length')
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.numpy()

    # Combine the text embeddings with the star rating
    combined_vector = np.concatenate(
        [embeddings.mean(axis=1).squeeze(), np.array([star_rating])])

    feature_vectors.append(combined_vector)

np.save('combined_feature_vectors.npy', np.array(feature_vectors))
