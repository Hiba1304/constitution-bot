import random
import json
import torch
import os
from model import NeuralNet
from functions import bag_of_words, tokenize
from datetime import datetime
import hashlib

class UserManager:
    def __init__(self, users_file="users.json"):
        self.users_file = users_file
        self.users_data = self.load_users()

    def load_users(self):
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_users(self):
        with open(self.users_file, 'w', encoding='utf-8') as f:
            json.dump(self.users_data, f, ensure_ascii=False, indent=2)

    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()

    def create_account(self, username, password, full_name):
        if username in self.users_data:
            return False, "Ce nom d'utilisateur existe déjà."

        self.users_data[username] = {
            "password": self.hash_password(password),
            "full_name": full_name,
            "created_at": datetime.now().isoformat(),
            "last_login": None,
            "conversation_history": []
        }
        self.save_users()
        return True, "Compte créé avec succès!"

    def login(self, username, password):
        if username not in self.users_data:
            return False, "Nom d'utilisateur incorrect."
        if self.users_data[username]["password"] != self.hash_password(password):
            return False, "Mot de passe incorrect."
        self.users_data[username]["last_login"] = datetime.now().isoformat()
        self.save_users()
        return True, "Connexion réussie!"

    def get_user_info(self, username):
        return self.users_data.get(username, None)

    def add_to_history(self, username, question, response):
        if username in self.users_data:
            self.users_data[username]["conversation_history"].append({
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "response": response
            })
            self.save_users()

    def get_history(self, username, limit=10):
        if username in self.users_data:
            history = self.users_data[username]["conversation_history"]
            return history[-limit:] if len(history) > limit else history
        return []

class ConstitutionBot:
    def __init__(self, model_filename, intents_filename):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.user_manager = UserManager()
        self.current_user = None

        with open(intents_filename, 'r', encoding='utf-8') as f:
            self.intents = json.load(f)

        data = torch.load(model_filename)
        self.input_size = data["input_size"]
        self.hidden_size = data["hidden_size"]
        self.output_size = data["output_size"]
        self.all_words = data["all_words"]
        self.tags = data["tags"]

        self.model = NeuralNet(self.input_size, self.hidden_size, self.output_size).to(self.device)
        self.model.load_state_dict(data["model_state"])
        self.model.eval()

        self.bot_name = "Assistant Constitution"
        self.confidence_threshold = 0.6

    def predict(self, sentence):
        sentence = tokenize(sentence)
        x = bag_of_words(sentence, self.all_words)
        x = torch.from_numpy(x).to(self.device).float().unsqueeze(0)
        output = self.model(x)
        _, predicted = torch.max(output, dim=1)
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        return self.tags[predicted.item()], prob.item()

    def get_response(self, message):
        tag, confidence = self.predict(message)
        if confidence > self.confidence_threshold:
            for intent in self.intents["intents"]:
                if tag == intent["tag"]:
                    return random.choice(intent["responses"])
        return "Je ne suis pas sûr de la réponse. Pouvez-vous reformuler ?"