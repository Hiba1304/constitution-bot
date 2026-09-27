import streamlit as st
from chat import ConstitutionBot
from PIL import Image
from datetime import datetime
import base64
import os

# --- Configuration des chemins (modifiable via variables d'environnement) ---
MODEL_PATH = os.environ.get("BOT_MODEL_PATH", "best_model.pth")
INTENTS_PATH = os.environ.get("BOT_INTENTS_PATH", "second_data_set.json")
AVATAR_PATH = os.environ.get("BOT_AVATAR_PATH", os.path.join("assets", "avatar.jpg"))

# Initialisation du bot
bot = ConstitutionBot(MODEL_PATH, INTENTS_PATH)

# Fonction pour convertir l'image en base64 pour l'afficher dans le HTML
def get_base64_image(img_path):
    if not os.path.exists(img_path):
        return None
    with open(img_path, "rb") as img_file:
        b64_str = base64.b64encode(img_file.read()).decode()
        return f"data:image/jpeg;base64,{b64_str}"

avatar_data_uri = get_base64_image(AVATAR_PATH)
avatar_html = f'<img src="{avatar_data_uri}" width="48" style="border-radius: 50%;">' if avatar_data_uri else "🤖"

# Configuration de la page
st.set_page_config(page_title="Assistant Constitutionnel", page_icon="⚖", layout="wide")

# Application du thème personnalisé via CSS
st.markdown("""
    <style>
        /* Fond global blanc */
        .stApp {
            background-color: #FFFFFF !important;
            color: #000000 !important;
            font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
        }

        /* Barre latérale avec fond blanc cassé */
        .sidebar .sidebar-content {
            background-color: #F8F9F9 !important;
            color: #000000 !important;
            border-right: 2px solid #E74C3C;
        }

        /* Boutons rouges avec texte blanc */
        .stButton>button {
            background-color: #E74C3C !important;
            color: #FFFFFF !important;
            border-radius: 8px;
            font-weight: bold;
            border: none;
            padding: 8px 16px;
        }
        .stButton>button:hover {
            background-color: #C0392B !important;
            cursor: pointer;
        }

        /* Input texte blanc avec texte noir */
        .stTextInput>div>div>input {
            background-color: #FFFFFF !important;
            color: #000000 !important;
            border: 2px solid #27AE60;
            border-radius: 6px;
            padding: 8px;
        }
        .stTextInput>div>div>input:focus {
            outline: none;
            border-color: #2ECC71;
            box-shadow: 0 0 5px #2ECC71;
        }

        /* Bulles réponses bot */
        .bot-response {
            background-color: #27AE60;
            color: #FFFFFF;
            padding: 12px;
            border-radius: 12px;
            margin-bottom: 8px;
            max-width: 80%;
            font-size: 1rem;
        }

        /* Bulles messages utilisateur */
        .user-message {
            background-color: #E74C3C;
            color: #FFFFFF;
            padding: 12px;
            border-radius: 12px;
            margin-bottom: 8px;
            max-width: 80%;
            font-size: 1rem;
            align-self: flex-end;
        }

        /* Sidebar history style */
        .sidebar .markdown-text-container {
            font-size: 0.9rem;
            color: #34495E;
        }
    </style>
""", unsafe_allow_html=True)

# Gestion de l'état de session
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.full_name = ""
    st.session_state.chat_history = []

# Barre latérale pour l'historique des conversations
st.sidebar.title("Historique")
if st.session_state.logged_in:
    history = bot.user_manager.get_history(st.session_state.username, limit=10)
    for i, conv in enumerate(reversed(history), 1):
        timestamp = datetime.fromisoformat(conv['timestamp']).strftime("%d/%m/%Y %H:%M")
        st.sidebar.markdown(f"{i}. {timestamp}")
        st.sidebar.markdown(f"*Q:* {conv['question']}")
        st.sidebar.markdown(f"*R:* {conv['response']}")
        st.sidebar.markdown("---")

# Authentification
if not st.session_state.logged_in:
    st.title("Connexion à l'Assistant Constitutionnel")

    tab1, tab2 = st.tabs(["Connexion", "Créer un compte"])

    with tab1:
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")
        if st.button("Se connecter"):
            success, msg = bot.user_manager.login(username, password)
            if success:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.full_name = bot.user_manager.get_user_info(username)["full_name"]
                st.success(msg)
            else:
                st.error(msg)

    with tab2:
        full_name = st.text_input("Nom complet")
        new_username = st.text_input("Nouveau nom d'utilisateur")
        new_password = st.text_input("Nouveau mot de passe", type="password")
        if st.button("Créer le compte"):
            success, msg = bot.user_manager.create_account(new_username, new_password, full_name)
            if success:
                st.success(msg)
            else:
                st.error(msg)

else:
    st.sidebar.success(f"Connecté en tant que : {st.session_state.full_name}")
    if st.sidebar.button("Se déconnecter"):
        st.session_state.logged_in = False
        st.session_state.chat_history = []
        st.rerun()

    st.title("Assistant Constitutionnel du Maroc")

    # Affichage de l'historique des messages
    for entry in st.session_state.chat_history:
        # Message utilisateur
        st.markdown(f"<div class='user-message'>{entry['question']}</div>", unsafe_allow_html=True)

        # Message bot avec avatar image
        st.markdown(f"""
        <div style="display: flex; align-items: flex-start; gap: 10px; margin-bottom: 10px;">
            {avatar_html}
            <div class="bot-response">{entry['response']}</div>
        </div>
        """, unsafe_allow_html=True)

    # Entrée utilisateur
    user_input = st.chat_input("Posez votre question...")
    if user_input:
        response = bot.get_response(user_input)
        st.session_state.chat_history.append({"question": user_input, "response": response})
        bot.user_manager.add_to_history(st.session_state.username, user_input, response)

        # Affiche message utilisateur immédiatement
        st.markdown(f"<div class='user-message'>{user_input}</div>", unsafe_allow_html=True)
        # Affiche réponse bot avec avatar
        st.markdown(f"""
        <div style="display: flex; align-items: flex-start; gap: 10px; margin-bottom: 10px;">
            {avatar_html}
            <div class="bot-response">{response}</div>
        </div>
        """, unsafe_allow_html=True)