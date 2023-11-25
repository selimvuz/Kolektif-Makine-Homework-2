import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Load the dataset with UTF-16 encoding
df = pd.read_csv("Datasets/TRdata_1.csv", encoding="utf-16")

# Split the dataset into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Define a custom dataset


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx]) + 1  # Shift labels to start from 0

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained(
    "ytu-ce-cosmos/turkish-small-bert-uncased")
model = BertForSequenceClassification.from_pretrained(
    "ytu-ce-cosmos/turkish-small-bert-uncased", num_labels=3
)

# Create datasets and dataloaders
train_dataset = SentimentDataset(
    train_df["Metinler"].values, train_df["Duygular"].values, tokenizer
)
val_dataset = SentimentDataset(
    val_df["Metinler"].values, val_df["Duygular"].values, tokenizer
)

train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=512, shuffle=False)

# Set up training parameters
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            val_preds.extend(preds)
            val_labels.extend(labels.cpu().numpy())

    val_accuracy = accuracy_score(val_labels, val_preds)
    print(
        f"Epoch {epoch + 1}/{num_epochs} - Validation Accuracy: {val_accuracy}")

# Save the fine-tuned model
model.save_pretrained("fine_tuned_sentiment_model_1")
tokenizer.save_pretrained("fine_tuned_sentiment_model_1")
