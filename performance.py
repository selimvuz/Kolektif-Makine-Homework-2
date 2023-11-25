import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re
import pandas as pd
from sklearn.metrics import accuracy_score


def preprocess_text(text):
    text = str(text)
    text = re.sub(r"[^a-zA-Z0-9ğüşıöçĞÜŞİÖÇ]", " ", text)
    text = text.lower()
    return text


def predict_sentiment_ensemble(models, tokenizers, text, max_length=128):
    votes = []

    for model, tokenizer in zip(models, tokenizers):
        model.eval()
        preprocessed_text = preprocess_text(text)

        # Tokenize and handle long texts by splitting into chunks
        tokens = tokenizer(
            preprocessed_text,
            add_special_tokens=True,
            return_tensors="pt",
        )

        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        # Handle long texts by truncating to max_length
        if input_ids.shape[1] > max_length:
            input_ids = input_ids[:, :max_length]
            attention_mask = attention_mask[:, :max_length]

        with torch.no_grad():
            logits = model(input_ids=input_ids,
                           attention_mask=attention_mask).logits

        predicted_class = torch.argmax(logits, dim=1).item()
        # Shift back to the original label range (-1, 0, 1)
        votes.append(predicted_class - 1)

    # Perform majority voting
    majority_vote = max(set(votes), key=votes.count)
    return majority_vote


def calculate_ensemble_accuracy(models, tokenizers, texts, labels, max_length=512):
    predictions = []

    for text, label in zip(texts, labels):
        # Predict sentiment using majority voting with text truncation
        sentiment = predict_sentiment_ensemble(
            models, tokenizers, text, max_length)
        predictions.append(sentiment)

    # Calculate ensemble accuracy
    ensemble_accuracy = accuracy_score(labels, predictions)
    return ensemble_accuracy


def main():
    # Load the fine-tuned models and individual tokenizers for ensemble
    model_paths = ["fine_tuned_sentiment_model_1",
                   "fine_tuned_sentiment_model_2", "fine_tuned_sentiment_model_3"]
    models = [BertForSequenceClassification.from_pretrained(
        path) for path in model_paths]

    tokenizer_paths = ["fine_tuned_sentiment_model_1",
                       "fine_tuned_sentiment_model_2", "fine_tuned_sentiment_model_3"]
    tokenizers = [BertTokenizer.from_pretrained(
        path) for path in tokenizer_paths]

    # Load your dataset for evaluation
    df_eval = pd.read_csv("Datasets/TRdata.csv", encoding="utf-16")
    eval_texts = df_eval["Metinler"].values
    eval_labels = df_eval["Duygular"].values

    print("Calculating ensemble accuracy...")

    # Set the desired max_length for text truncation
    max_length = 512

    # Calculate ensemble accuracy with text truncation
    ensemble_accuracy = calculate_ensemble_accuracy(
        models, tokenizers, eval_texts, eval_labels, max_length)

    print(f"Ensemble Model Accuracy: {ensemble_accuracy:.4f}")


if __name__ == "__main__":
    main()
