import re
from pathlib import Path

import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from spacy.lang.fr.stop_words import STOP_WORDS

DATA_PATH = Path("data/dataset_nlp_test_tal.csv")


def pretraiter(texte, stop_fr):
    """Même prétraitement que dans le notebook."""
    texte = texte.lower()
    texte = re.sub(r"[^\w\s]", " ", texte)
    texte = re.sub(r"\d+", " ", texte)
    tokens = word_tokenize(texte, language="french")
    tokens = [t.lower() for t in tokens if t.isalpha() and len(t) > 2]
    return [t for t in tokens if t not in stop_fr]


@st.cache_resource(show_spinner="Entraînement du modèle...")
def charger_modele():
    for ressource in ("punkt_tab", "stopwords"):
        nltk.download(ressource, quiet=True)

    stop_fr = set(stopwords.words("french")) | set(STOP_WORDS)

    df = pd.read_csv(DATA_PATH)
    textes = [" ".join(pretraiter(t, stop_fr)) for t in df["texte"]]

    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(textes)

    svm = LinearSVC(max_iter=10000, random_state=42)
    svm.fit(X, df["categorie"])
    return tfidf, svm, stop_fr


def predire(texte, tfidf, svm, stop_fr):
    tokens = pretraiter(texte, stop_fr)
    X = tfidf.transform([" ".join(tokens)])
    scores = svm.decision_function(X)[0]
    return tokens, pd.Series(scores, index=svm.classes_)


st.set_page_config(page_title="Classification d'avis citoyens", layout="centered")

st.title("Classification d'avis citoyens")
st.write(
    "Saisissez un commentaire sur un service public : le modèle indique s'il "
    "relève d'une satisfaction, d'une insatisfaction ou d'une suggestion."
)

if not DATA_PATH.exists():
    st.error(f"Fichier introuvable : {DATA_PATH}. Lancez l'application depuis la racine du dépôt.")
    st.stop()

tfidf, svm, stop_fr = charger_modele()

texte = st.text_area(
    "Commentaire",
    height=120,
    placeholder="Ex. : Le personnel de la mairie est accueillant et le dossier a été traité rapidement.",
)

if st.button("Classer", type="primary"):
    if not texte.strip():
        st.warning("Veuillez saisir un texte.")
    else:
        tokens, scores = predire(texte, tfidf, svm, stop_fr)

        if not tokens:
            st.warning(
                "Aucun mot exploitable après prétraitement (mots vides, "
                "mots de moins de 3 lettres). La prédiction n'est pas fiable."
            )

        st.subheader(f"Catégorie prédite : {scores.idxmax()}")

        if scores.max() < 0:
            st.info("Tous les scores sont négatifs : aucune classe ne se détache nettement.")

        st.bar_chart(scores.rename("score"))
        st.caption(
            "Les scores sont les marges du SVM (decision_function), pas des "
            "probabilités. Plus le score est élevé, plus le modèle penche pour la classe."
        )

        with st.expander("Mots retenus après prétraitement"):
            st.write(", ".join(tokens) if tokens else "(aucun)")

st.divider()
st.caption(
    "Modèle : TF-IDF + SVM linéaire, entraîné sur les 150 avis du jeu de données. "
    "Accuracy d'environ 0,66 en validation croisée. Les mots en éwé ne sont pas traités."
)
