import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel, AdamW
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split

class DistilBERT_Model(nn.Module):
    def __init__(self, num_labels):
        super(DistilBERT_Model, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.distilbert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # Take the CLS token
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output)
        return logits

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128, is_test=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['Text']
        if not self.is_test:
            label = self.data.iloc[idx]['Label']
        else:
            label = 0  # Placeholder label for test dataset
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Load train and test datasets
# train_data = pd.read_csv('train.csv')
# test_data = pd.read_csv('test.csv')

# Split train data into train and validation sets
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

# Initialize tokenizer and model (use DistilBERT)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBERT_Model(num_labels=2)

# Define datasets and data loaders for train, validation, and test
train_dataset = CustomDataset(train_data, tokenizer)
val_dataset = CustomDataset(val_data, tokenizer)
test_dataset = CustomDataset(test_data, tokenizer, is_test=True)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)  # Compute the loss

        loss.backward()
        optimizer.step()

    # Validation evaluation after each epoch
    model.eval()
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_accuracy = correct_val / total_val
    print(f'Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {val_accuracy:.4f}')

# Evaluation on test data
model.eval()
test_predictions = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        _, predicted = torch.max(outputs, 1)
        test_predictions.extend(predicted.cpu().detach().numpy())

# Convert predictions to DataFrame with 'ID' column
test_ids = test_data['ID']
predictions_df = pd.DataFrame({'ID': test_ids, 'Label': test_predictions})

# Save predictions to CSV
predictions_df.to_csv('distilbert_nn2.csv', index=False)
