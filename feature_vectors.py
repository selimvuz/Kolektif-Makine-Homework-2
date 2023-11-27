from transformers import BertTokenizer, TFBertModel
import numpy as np
import pandas as pd
import json

# JSON verilerini doğrudan yükleyin
dataset = pd.read_json('Datasets/part_0.json')

tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-uncased')
model = TFBertModel.from_pretrained('dbmdz/bert-base-turkish-uncased')

# Öznitelik vektörlerini depolamak için bir liste oluşturun
feature_vectors = []
max_sequence_length = 512

for index, row in dataset.iterrows():
    text = row['text'][:max_sequence_length - 2]
    star_rating = row['star']

    # Metni tokenize edin ve modeli kullanarak gömme vektörünü alın
    inputs = tokenizer(text, return_tensors="tf", truncation=True,
                       max_length=max_sequence_length, padding='max_length')
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.numpy()

    # Metin gömme vektörlerini ortalayın ve yıldız derecelendirmesini ekleyin
    combined_vector = np.concatenate(
        [embeddings.mean(axis=1).squeeze(), np.array([star_rating])])

    feature_vectors.append(combined_vector)

np.save('combined_feature_vectors.npy', np.array(feature_vectors))
