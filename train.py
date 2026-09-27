import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from functions import tokenize, stem, bag_of_words
from model import NeuralNet

# Charger les données
with open("dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

all_words = []
tags = []
xy = []
ignore_words = ['?', '!', '.', ',']

# Prétraitement des phrases
for intent in data['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

all_words = sorted(set(stem(w) for w in all_words if w not in ignore_words))
tags = sorted(set(tags))

X_train = []
Y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    Y_train.append(label)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

# Dataset personnalisé
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train

    def __getitem__(self, index):
        return torch.tensor(self.x_data[index], dtype=torch.float32), torch.tensor(self.y_data[index], dtype=torch.long)

    def __len__(self):
        return self.n_samples

# Hyperparamètres
batch_size = 8
input_size = len(X_train[0])
hidden_size = 16
output_size = len(tags)
learning_rate = 0.001
num_epochs = 1000

# DataLoader
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# Initialisation du modèle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Fonction de perte et optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Entraînement
for epoch in range(num_epochs):
    for words, labels in train_loader:
        words, labels = words.to(device), labels.to(device)

        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print(f"Final loss: {loss.item():.4f}")

# Sauvegarde du modèle
model_data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

torch.save(model_data, "model.pth")
print("Training complete. File saved to model.pth")
