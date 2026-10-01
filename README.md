# Classification de commentaires citoyens sur les services publics (NLP)

Test technique – Traitement du langage naturel, sélection TAISS 2026 / Afriklang (Togo Data Lab).
Auteur : *TENGA Cherif Abdel Azize - Data Scientist & AI Engineer*

---

## Résumé (approche, choix et résultats)

- **Objectif** : classer 150 commentaires citoyens en *Satisfaction*, *Insatisfaction* ou *Suggestion* (50 par classe, textes d'environ 9 à 10 mots).
- **Prétraitement** : minuscules, suppression de la ponctuation et des chiffres, tokenisation NLTK, stopwords français (union NLTK + spaCy, 579 mots).
- **Éwé / mina** : trois stratégies analysées (suppression, traduction, lexique de substitution), mais aucune n'est implémentée : ces mots restent dans le texte (limite identifiée).
- **Vectorisation** : TF-IDF retenu (457 termes). Bag of Words et Word2Vec ont été explorés, sans être utilisés pour la classification.
- **Modèles** : Naive Bayes, Régression logistique, SVM linéaire ; découpage 80/20 avec `random_state=42`.
- **Résultats sur le jeu de test (30 avis)** : accuracy 0,47 / 0,53 / 0,53 et F1 macro 0,47 / 0,53 / 0,53 (Naive Bayes / Régression logistique / SVM).
- **Validation croisée 5 plis (SVM, 150 avis)** : accuracy 0,66 et F1 macro 0,66, estimation plus fiable qu'un test de 30 avis.
- **Erreurs** : 78 % des erreurs impliquent *Insatisfaction* ; la paire Insatisfaction ↔ Suggestion est la plus confondue (22 erreurs sur 51).
- **Causes probables** : négations, vocabulaire thématique commun aux classes, *Suggestion* définie par une intention plutôt qu'un sentiment, textes très courts.
- **Limites** : très peu de données, TF-IDF sans sémantique, éwé non traité, évaluation à forte variance.
- **Pistes** : modèle d'embeddings multilingue pré-entraîné (SetFit ou fine-tuning), lexique éwé, conservation des négations et n-grammes, clarification des consignes d'étiquetage.

---

## 1. Structure du dépôt

```
.
├── README.md
├── requirements.txt
├── code.ipynb                      # notebook de travail (toutes les étapes)
├── data/
│   └── dataset_nlp_test_tal.csv    # 150 commentaires (id, texte, categorie)
└── figures_dir/                    # figures générées (wordclouds, matrices de confusion, comparaison)
```

## 2. Installation et reproduction

```bash
python -m venv .venv
source .venv/bin/activate            # Windows : .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download fr_core_news_sm
jupyter notebook code.ipynb
```

- Bibliothèques principales : `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `nltk`, `spacy`, `wordcloud`, `gensim`.
- Les ressources NLTK (`punkt_tab`, `stopwords`) sont téléchargées par le notebook.
- Avant de relancer, adapter la variable `data_path` (chemin du CSV) et `figures_dir` à votre machine.
- Graines fixées : `random_state=42` pour le découpage train/test, `random_state=0` pour la validation croisée.

---

## 3. Données et exploration

- **Dimensions** : 150 lignes × 3 colonnes (`id`, `texte`, `categorie`), sans valeur manquante.
- **Classes** : parfaitement équilibrées, 50 avis par catégorie. L'accuracy et le F1 macro sont donc comparables.
- **Longueur moyenne** (en mots) : Insatisfaction 9,34 ; Satisfaction 8,68 ; Suggestion 10,50. L'écart entre classes est faible (environ 2 mots), la longueur est donc un indice peu discriminant.

### Termes les plus fréquents (après nettoyage)

| Classe | Termes dominants (occurrences) | Lecture |
|---|---|---|
| Insatisfaction | aucune (10), dossier (6), service (5), personnel, attente, mois, manque (4) | Vocabulaire de l'absence et du manque |
| Suggestion | créer (6), ligne, système, mettre, services, agents (5), prévoir, proposer (4) | Verbes d'action à l'infinitif, signe d'une intention |
| Satisfaction | service (14), bien (8), bonne (5), satisfait, rapidement (4) | Adjectifs et adverbes d'évaluation positive |

Le mot « service » apparaît dans les trois classes, ce qui annonce une partie des confusions analysées plus bas.
Les nuages de mots par catégorie et les histogrammes de fréquence sont enregistrés dans `figures_dir/`.

---

## 4. Prétraitement

1. **Minuscules**, puis suppression de la **ponctuation, des caractères spéciaux et des chiffres** par expressions régulières. Les accents sont conservés (utiles en français et en éwé).
2. **Tokenisation** avec `nltk.word_tokenize` (langue française), en gardant les tokens alphabétiques de plus de 2 lettres.
3. **Stopwords** : union des listes NLTK (157 mots) et spaCy (507 mots), soit 579 mots uniques.
   *Justification* : une union est plus exhaustive qu'une liste seule, ce qui réduit le bruit sur un corpus où chaque mot compte.
4. **Mots en éwé / mina** : trois options ont été comparées :
   - suppression : simple, mais elle efface parfois le seul indice de polarité (ex. « Yèvu service la nyuie hafi! », classé *Satisfaction*) ;
   - traduction automatique : peu adaptée à un texte qui mélange deux langues ;
   - substitution par un lexique : la plus fiable, mais elle demande de construire et de faire valider le lexique.
   **Aucune n'a été appliquée à ce stade** : les mots éwé restent dans le texte comme des tokens ordinaires.

---

## 5. Modélisation

### Vectorisation

| Méthode | Statut | Remarque |
|---|---|---|
| **TF-IDF** (`max_features=5000`, 457 termes effectifs) | **Retenue** | Rapide, adaptée aux petits corpus, pondère les mots rares et distinctifs |
| Bag of Words | Explorée | Même vocabulaire (457 termes), sans pondération |
| Word2Vec (CBOW, 100 dimensions) | Explorée | Vocabulaire de seulement 136 mots (`min_count=2`) |

*Pourquoi TF-IDF et pas Word2Vec* : un Word2Vec entraîné sur 150 textes d'une dizaine de mots n'a pas assez de contexte pour apprendre des représentations de qualité. Les embeddings utiles viendraient d'un modèle **pré-entraîné** (voir pistes d'amélioration).

### Entraînement

- Découpage 80 % / 20 % (`random_state=42`) : 120 avis d'entraînement, 30 de test. Le découpage n'est pas stratifié, d'où un test déséquilibré (11 Insatisfaction, 12 Satisfaction, 7 Suggestion).
- Trois algorithmes : **Naive Bayes multinomial**, **Régression logistique** (`max_iter=1000`), **SVM linéaire** (`LinearSVC`).

---

## 6. Résultats

### 6.1 Jeu de test (30 avis)

| Modèle | Accuracy | F1 macro | F1 pondéré |
|---|---|---|---|
| Naive Bayes | 0,467 | 0,474 | 0,458 |
| Régression logistique | 0,533 | 0,533 | 0,523 |
| SVM linéaire | 0,533 | 0,534 | 0,531 |

**Lecture** : la régression logistique et le SVM sont à égalité (écart de 0,0004 sur le F1 macro, sans signification), et tous deux devancent Naive Bayes. Le SVM est retenu par convention pour la suite.

Sur 30 avis, une seule erreur change l'accuracy de plus de 3 points ; l'incertitude est d'environ **±18 points** (intervalle binomial approximatif). Ces trois scores ne permettent donc pas de départager sérieusement les modèles.

### 6.2 Validation croisée stratifiée à 5 plis (SVM, 150 avis)

Chacun des 150 avis est prédit par un modèle qui ne l'a pas vu pendant l'entraînement.

| Indicateur | Valeur |
|---|---|
| Accuracy | **0,66** (99 / 150), soit environ ±8 points |
| F1 macro | **0,66** |

| Classe | Précision | Rappel | F1 |
|---|---|---|---|
| Insatisfaction | 0,61 | 0,54 | 0,57 |
| Satisfaction | 0,70 | 0,74 | 0,72 |
| Suggestion | 0,66 | 0,70 | 0,68 |

**Matrice de confusion** (lignes = classe réelle, colonnes = classe prédite) :

| Réel ↓ / Prédit → | Insatisfaction | Satisfaction | Suggestion |
|---|---|---|---|
| **Insatisfaction** | **27** | 10 | 13 |
| **Satisfaction** | 8 | **37** | 5 |
| **Suggestion** | 9 | 6 | **35** |

**Pourquoi 0,66 en validation croisée et 0,53 sur le test ?** Les deux évaluations entraînent sur 120 avis. L'écart vient donc surtout du bruit d'un test de 30 avis (et d'un découpage non stratifié), pas d'une différence de modèle. C'est la validation croisée qui doit servir de référence. Seul le SVM a été évalué de cette façon.

---

## 7. Analyse des erreurs

### 7.1 Vue d'ensemble (51 erreurs en validation croisée)

| Paire confondue | Erreurs | Détail |
|---|---|---|
| Insatisfaction ↔ Suggestion | **22** | 13 + 9 |
| Insatisfaction ↔ Satisfaction | **18** | 10 + 8 |
| Satisfaction ↔ Suggestion | **11** | 5 + 6 |

1. **L'Insatisfaction est au centre des erreurs** : 40 erreurs sur 51 (78 %) l'impliquent, avec le rappel le plus bas (0,54).
2. **La Satisfaction est la mieux reconnue** (F1 = 0,72), probablement grâce à un vocabulaire d'évaluation très marqué (« bien », « bonne », « satisfait »).
3. **Les confusions vont dans les deux sens** (13 Insatisfaction → Suggestion contre 9 Suggestion → Insatisfaction). Cela suggère un **recouvrement réel** entre classes plutôt qu'un biais du modèle. Les écarts entre sens sont trop faibles, à 50 exemples par classe, pour être interprétés.

### 7.2 Cinq exemples mal classés

Les textes sont affichés après prétraitement. Le « score » est la valeur de `decision_function` du SVM (une marge, pas une probabilité) : une valeur négative signifie qu'aucune classe ne l'emporte nettement. Les causes ci-dessous sont des **hypothèses**, non vérifiées par une expérience de contrôle.

| # | Avis (prétraité) | Réel → Prédit | Score | Hypothèse sur la cause |
|---|---|---|---|---|
| 80 | formations régulières personnel amélioreraient qualité service | Suggestion → Satisfaction | 0,23 | « qualité » et « service » dominent le vocabulaire de la Satisfaction. Le seul marqueur de suggestion est le conditionnel « amélioreraient », forme rare que TF-IDF ne généralise pas (pas de lemmatisation). |
| 143 | système prise rendez ligne pratique | Satisfaction → Suggestion | 0,20 | « système », « ligne » et « rendez » sont des termes fréquents des Suggestions (thème des démarches en ligne). Le modèle suit le **sujet** et non le **sentiment**. |
| 25 | publier résultats enquêtes satisfaction transparence | Suggestion → Insatisfaction | −0,29 | Score négatif : le modèle n'avait pas de signal net. Les mots sont rares dans le corpus, et l'infinitif « publier » n'a pas été appris comme marqueur de suggestion. |
| 10 | site web inaccessible jours | Insatisfaction → Suggestion | 0,15 | Seulement 4 tokens après prétraitement, donc très peu de signal. « site » et « jours » existent dans les deux classes. |
| 74 | service classe aucune remarque négative | Satisfaction → Insatisfaction | 0,16 | **Double négation** : « aucune » (premier terme de l'Insatisfaction) et « négative » orientent vers l'Insatisfaction, alors que l'avis est positif. Un modèle de mots isolés ne capte pas ce renversement. |

### 7.3 Synthèse des causes

- **Négation et ordre des mots** perdus par le sac de mots (exemple 74).
- **Vocabulaire thématique partagé** : les classes parlent des mêmes sujets (service, dossier, ligne), seul le ton change (exemples 80 et 143).
- ***Suggestion* repose sur une intention** (proposer, demander) et non sur un sentiment, ce qui la fait recouvrir les deux autres classes : une suggestion naît souvent d'une insatisfaction.
- **Textes courts après nettoyage** : 4 à 6 tokens suffisent à rendre la décision fragile (exemples 10 et 25).
- **Étiquetage possiblement ambigu** pour les avis qui critiquent *et* proposent : à vérifier par une relecture des erreurs Insatisfaction ↔ Suggestion.

---

## 8. Conclusion

Un pipeline complet (exploration, nettoyage, TF-IDF, trois classifieurs, évaluation, analyse d'erreurs) a été construit. Les performances restent **modestes** : environ 66 % d'accuracy en validation croisée, soit le double du hasard (33 %). Les trois classifieurs obtiennent des résultats proches, ce qui indique que le **facteur limitant est la représentation du texte (et la taille des données), non le choix de l'algorithme**. L'analyse d'erreurs confirme que les échecs viennent surtout de la négation, du vocabulaire partagé entre classes et du caractère « intention » de la classe Suggestion.

### Limites

- **Très peu de données** : 150 avis, un test de 30 avis (±18 points), une validation croisée à ±8 points. Un écart de 1 à 2 points de F1 n'est pas interprétable.
- **TF-IDF ajusté sur tout le jeu avant le découpage** : le vocabulaire et les pondérations IDF ont vu les avis de test. Le biais est probablement faible mais réel ; un `Pipeline` scikit-learn l'éliminerait.
- **Négations potentiellement supprimées** : les listes de stopwords standard contiennent généralement « ne » et « pas », et la contrainte de plus de 2 lettres élimine d'autres petits mots porteurs de sens.
- **Éwé / mina non traités** : ces mots, parfois seuls porteurs de la polarité, sont ignorés ou mal vectorisés.
- **Comparaison de modèles sur un seul test de 30 avis** : seul le SVM a été évalué en validation croisée.
- **Analyse d'erreurs sur 5 cas** : un échantillon trop petit pour généraliser, les causes restent des hypothèses.

### Pistes d'amélioration

1. **Embeddings pré-entraînés multilingues** (ex. `paraphrase-multilingual-MiniLM-L12-v2`, `multilingual-e5-small`, ou CamemBERT), puis **SetFit** ou fine-tuning léger. Ces modèles ont déjà appris le sens des phrases et la négation, ce qui répond directement aux causes identifiées. À 50 exemples par classe, SetFit est plus robuste qu'un fine-tuning complet. Une évaluation par validation croisée est indispensable.
2. **Lexique éwé / mina validé par un locuteur**, avec détection mot à mot (caractères propres à l'éwé, mots hors vocabulaire français) puis substitution par l'équivalent français, pour que ces mots portent leur polarité.
3. **Mieux traiter la négation** : retirer « ne », « pas », « aucune », « sans » de la liste de stopwords, utiliser des n-grammes (1-2) et une lemmatisation spaCy.
4. **Clarifier les classes** : relire les erreurs Insatisfaction ↔ Suggestion, rédiger une règle d'étiquetage pour les avis mixtes, voire passer en classification multi-étiquette.
5. **Évaluation plus rigoureuse** : validation croisée répétée pour tous les modèles, `Pipeline` pour éviter la fuite de données, stratification du découpage.
6. **Bonus possibles** : interface de démonstration (Streamlit ou Gradio) et gestion explicite des commentaires mixtes français / éwé-mina.