import torch
import torch.nn as nn
import json
from model import NeuralNet
from functions import bag_of_words, tokenize

class ConstitutionBot:
    def __init__(self, model_path, data_path):
        # Charger le fichier JSON contenant les intents
        with open("second_data_set.json", 'r', encoding='utf-8') as f:
            self.intents = json.load(f)

        # Charger le modèle entraîné
        model_data = torch.load(model_path)
        self.all_words = model_data['all_words']
        self.tags = model_data['tags']
        input_size = model_data['input_size']
        hidden_size = model_data['hidden_size']
        output_size = model_data['output_size']

        self.model = NeuralNet(input_size, hidden_size, output_size)
        self.model.load_state_dict(model_data['model_state'])
        self.model.eval()

    def get_response(self, message):
        sentence = tokenize(message)
        X = bag_of_words(sentence, self.all_words)
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(0)

        output = self.model(X)
        _, predicted = torch.max(output, dim=1)
        tag = self.tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.75:
            for intent in self.intents['intents']:
                if tag == intent['tag']:
                    return intent['responses'][0]
        return "Désolé, je ne comprends pas votre question."