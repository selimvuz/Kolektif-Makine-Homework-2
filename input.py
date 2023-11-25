import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re


def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z0-9ğüşıöçĞÜŞİÖÇ]", " ", text)
    text = text.lower()
    return text


def predict_sentiment(models, tokenizers, text):
    votes = []

    for model, tokenizer in zip(models, tokenizers):
        model.eval()
        preprocessed_text = preprocess_text(text)
        inputs = tokenizer(preprocessed_text, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_class = torch.argmax(logits, dim=1).item()
        # Shift back to the original label range (-1, 0, 1)
        votes.append(predicted_class - 1)

    # Perform majority voting
    majority_vote = max(set(votes), key=votes.count)
    return majority_vote


def main():
    # Load the fine-tuned models and individual tokenizers
    model_paths = ["fine_tuned_sentiment_model_1",
                   "fine_tuned_sentiment_model_2", "fine_tuned_sentiment_model_3"]
    models = [BertForSequenceClassification.from_pretrained(
        path) for path in model_paths]

    tokenizer_paths = ["fine_tuned_sentiment_model_1",
                       "fine_tuned_sentiment_model_2", "fine_tuned_sentiment_model_3"]
    tokenizers = [BertTokenizer.from_pretrained(
        path) for path in tokenizer_paths]

    # Get user input
    user_input = input("Enter a text for sentiment analysis: ")

    # Predict sentiment using majority voting
    sentiment = predict_sentiment(models, tokenizers, user_input)

    # Map sentiment to human-readable label
    sentiment_mapping = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
    sentiment_label = sentiment_mapping[sentiment]

    print(f"Predicted sentiment: {sentiment_label}")


if __name__ == "__main__":
    main()
