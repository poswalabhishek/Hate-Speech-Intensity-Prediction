import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random
import os

# Set the seed for reproducibility
def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed = 42
random_seed(seed)

# Check if CUDA is available and set PyTorch to use GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Load the RoBERTa model
model = RobertaModel.from_pretrained('roberta-base')

# Set the model to use GPU or CPU
model = model.to(device)

# Define the parameters
max_length = 256
batch_size = 32
epochs = 30

# Load the data
base_folder = "../../datasets/"
input_file = "hate_int_prof_SVO.tsv"
df = pd.read_csv(os.path.join(base_folder, input_file), delimiter='\t')

# Split the data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(df['Sentence'], df['Intensity'], test_size=0.2, random_state=seed)

# Tokenize the data
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=max_length)
val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True, max_length=max_length)

# Define a custom dataset
class HateSpeechDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create the data loaders
train_dataset = HateSpeechDataset(train_encodings, train_labels.tolist())
val_dataset = HateSpeechDataset(val_encodings, val_labels.tolist())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Train the model
model.train()
for epoch in range(epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()