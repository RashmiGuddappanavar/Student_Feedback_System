import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ========== Step 1: Load and Preprocess Dataset ==========

# Load the dataset
data = pd.read_csv(r'dataset/trainingdata.csv')  # Ensure CSV has 'sentiments' and 'sentences' columns

# Handle missing labels by removing them
data = data.dropna(subset=['sentiments'])

# Ensure sentiments are integers
data['sentiments'] = data['sentiments'].astype(int)

# Print unique sentiments in the dataset
print("Unique Sentiments in the Dataset:", data['sentiments'].unique())

# Verify for invalid labels
print("Rows with invalid labels:\n", data[data['sentiments'].isna()])

# Define sentiment mapping (adjust if required)
label_mapping = {-1: 0, 0: 1, 1: 2}
original_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}  # Reverse mapping

# Map sentiments to a range starting at 0
data['sentiments'] = data['sentiments'].map(label_mapping)

# ========== Step 2: Define Dataset Class ==========

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ========== Step 3: Split Dataset and Create DataLoaders ==========

# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    data['sentences'], data['sentiments'], test_size=0.1, random_state=42
)

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define maximum sequence length for BERT
MAX_LEN = 128

# Create DataLoader for training and validation
train_dataset = SentimentDataset(X_train.to_numpy(), y_train.to_numpy(), tokenizer, MAX_LEN)
val_dataset = SentimentDataset(X_val.to_numpy(), y_val.to_numpy(), tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# ========== Step 4: Initialize Model, Optimizer, and Scheduler ==========

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
model = model.to('cuda')  # Move model to GPU if available

# Define optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# ========== Step 5: Training the Model ==========

EPOCHS = 3
for epoch in range(EPOCHS):
    model.train()
    loop = tqdm(train_loader, leave=True)
    total_loss = 0

    for batch in loop:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to('cuda')
        attention_mask = batch['attention_mask'].to('cuda')
        labels = batch['labels'].to('cuda')

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update progress bar
        loop.set_description(f'Epoch {epoch + 1}')
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch + 1} completed. Average loss: {total_loss / len(train_loader):.4f}")

# ========== Step 6: Evaluate the Model ==========

model.eval()
predictions, true_labels = [], []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to('cuda')
        attention_mask = batch['attention_mask'].to('cuda')
        labels = batch['labels'].to('cuda')

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Get predicted labels
        preds = torch.argmax(logits, dim=1)

        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Print accuracy and classification report
accuracy = accuracy_score(true_labels, predictions)
print(f"Accuracy Score: {accuracy:.4f}")
print("Classification Report:\n", classification_report(true_labels, predictions))

# Map predictions back to original sentiments
preds_mapped = [original_mapping[pred] for pred in predictions]

# Show first few predictions
print("Predictions (Mapped to Original Sentiments):", preds_mapped[:10])
 