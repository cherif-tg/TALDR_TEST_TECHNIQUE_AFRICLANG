import re
from pathlib import Path

import joblib
import nltk
import pandas as pd
import streamlit as st
from nltk.tokenize import word_tokenize

MODEL_PATH = Path("models/modele_final.joblib")


def pretraiter(texte, stop_fr):
    """Même prétraitement que dans le notebook."""
    texte = texte.lower()
    texte = re.sub(r"[^\w\s]", " ", texte) # Suppression des caractères spéciaux
    texte = re.sub(r"\d+", " ", texte) # Suppression des chiffres
    tokens = word_tokenize(texte, language="french") # tokenisation
    tokens = [t.lower() for t in tokens if t.isalpha() and len(t) > 2]
    return [t for t in tokens if t not in stop_fr]


@st.cache_resource(show_spinner="Chargement du modèle...")
def charger_modele():
    nltk.download("punkt_tab", quiet=True)
    bundle = joblib.load(MODEL_PATH) # Chargement du modele et du vectorizer sauvegarder via joblib
    return bundle["tfidf"], bundle["modele"], set(bundle["stop_fr"]), bundle.get("nom", "modèle")


def predire(texte, tfidf, modele, stop_fr):
    """Fonction de prediction qui utilise le modele sauvegarder pour faire
    Une classification du texte saisi par l'utilisateur."""
    tokens = pretraiter(texte, stop_fr)
    X = tfidf.transform([" ".join(tokens)])
    if hasattr(modele, "decision_function"):
        scores = modele.decision_function(X)[0]
    else:  # Naive Bayes n'a pas de decision_function
        scores = modele.predict_proba(X)[0]
    return tokens, pd.Series(scores, index=modele.classes_)


st.set_page_config(page_title="Classification d'avis citoyens", layout="centered")

st.title("Classification d'avis citoyens")
st.write(
    "Saisissez un commentaire sur un service public : le modèle indique s'il "
    "relève d'une satisfaction, d'une insatisfaction ou d'une suggestion."
)

if not MODEL_PATH.exists():
    st.error(f"Modèle introuvable : {MODEL_PATH}. Exécutez d'abord la sauvegarde depuis le notebook.")
    st.stop()

tfidf, modele, stop_fr, nom = charger_modele()

texte = st.text_area(
    "Commentaire",
    height=120,
    placeholder="Ex. : Le personnel de la mairie est accueillant et le dossier a été traité rapidement.",
)

if st.button("Classer", type="primary"):
    if not texte.strip():
        st.warning("Veuillez saisir un texte.")
    else:
        tokens, scores = predire(texte, tfidf, modele, stop_fr)

        if not tokens:
            st.warning(
                "Aucun mot exploitable après prétraitement (mots vides, "
                "mots de moins de 3 lettres). La prédiction n'est pas fiable."
            )

        st.subheader(f"Catégorie prédite : {scores.idxmax()}")

        if hasattr(modele, "decision_function") and scores.max() < 0:
            st.info("Tous les scores sont négatifs : aucune classe ne se détache nettement.")

        st.bar_chart(scores.rename("score"))

        with st.expander("Mots retenus après prétraitement"):
            st.write(", ".join(tokens) if tokens else "(aucun)")

st.divider()
st.caption(f"Modèle utilisé : {nom} (vectorisation TF-IDF). Les mots en éwé ne sont pas traités.")