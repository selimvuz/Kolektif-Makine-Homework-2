from transformers import pipeline

# Zero-shot sınıflandırma modelini yükle
pipe = pipeline("zero-shot-classification")

# Metni sınıflandır
result = pipe("I have a problem with my iPhone that needs to be resolved asap!",
              candidate_labels=["urgent", "not urgent", "phone"])

# Sonucu yazdır
print("Predicted Label:", result["labels"][0])
print("Confidence Score:", result["scores"][0])
