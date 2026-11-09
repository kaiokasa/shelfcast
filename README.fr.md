# 🛒 ShelfCast

**Prévision explicable des ventes Walmart** : des prévisions de ventes hebdomadaires pour 45 magasins, expliquées en langage clair par un LLM local.

[English](README.md) | **Français**

[![CI](https://github.com/kaiokasa/shelfcast/actions/workflows/ci.yml/badge.svg)](https://github.com/kaiokasa/shelfcast/actions/workflows/ci.yml)

> - **Objectif :** prévoir les ventes hebdomadaires de 3 169 couples magasin/département dans 45 magasins Walmart, jusqu'à 39 semaines à l'avance ([compétition Kaggle](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting)).
> - **Résultat :** **WMAE de 1 818** sur une année de validation qui inclut Thanksgiving et Noël (XGBoost de référence : 2 752, sur une validation plus facile).
> - **Kaggle :** WMAE privé de **2 885**, public de **2 769** (≈ 78e sur 689 au classement public, top 11 % ; soumission tardive, non classée officiellement). L'écart avec notre validation vient très probablement de conditions du test qu'elle ne couvrait pas : Noël 2012 tombe un mardi (jamais vu à l'entraînement) et un tiers des lignes n'ont pas de CPI ni de chômage. La période du calendrier, elle, explique peu de choses (notre validation restreinte à la même période novembre → juillet passe seulement de 1 818 à 1 889).
> - **Modèle :** 50 % LightGBM avec des variables de décalage de 52 semaines + 50 % prévision « même semaine l'an dernier », débruitée entre magasins (SVD).
> - **Idées clés :** une validation qui contient les pics des fêtes, des décalages adaptés à l'horizon de prévision, et la combinaison d'un modèle d'arbres avec une prévision saisonnière simple.
> - **Explorer :** [`Notebook.ipynb`](Notebook.ipynb) (le projet étape par étape), [`Notebook_Experiments.ipynb`](Notebook_Experiments.ipynb) (ce qui a été testé), application Streamlit : `streamlit run app.py`.

---

## 🏆 Modèle final : combinaison LightGBM + prévision naïve saisonnière

**Prédiction = 50 % × LightGBM avec variables de décalage + 50 % × prévision naïve saisonnière** (« même semaine l'an dernier », sur un historique débruité par SVD entre magasins, rang 10).

Validation : novembre 2011 → octobre 2012 (52 semaines, **avec Thanksgiving et Noël**) :

| Modèle | WMAE | MAE | R² | MAE Thanksgiving et Noël |
| --- | --- | --- | --- | --- |
| LightGBM + décalages | 2 068 | 1 960 | 0,9637 | 2 988 |
| Naïve saisonnière (SVD rang 10) | 1 887 | 1 813 | 0,9703 | 2 500 |
| **Moyenne 50/50** | **1 818** | **1 764** | **0,9722** | **2 285** |

* Les deux modèles ne font pas les mêmes erreurs : LightGBM surestime les semaines avant Noël, alors que la prévision naïve suit de près le schéma de l'année précédente. Leur moyenne bat les deux.
* Le poids 50/50 a été fixé à l'avance, pas ajusté sur la validation.
* Le modèle est validé sur la période qui inclut les fêtes, puis **réentraîné sur toutes les données (2010 → 2012)** pour l'application et la soumission.
* Variables de LightGBM = variables de base + `Sales_Lag_52`, `Sales_Lag_52_Mean4`, `Store_Dept_Mean`, `Store_Dept_Median`. Toute la logique se trouve dans `walmart_features.py`, partagé par le notebook et l'application.
* LightGBM s'entraîne sur CPU en environ 10 secondes, et la SVD prend environ 1 seconde.

```python
LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, num_leaves=63,
              subsample=0.8, colsample_bytree=0.8, min_child_samples=20, random_state=42)
```

---

## 🚀 Application Streamlit

Une application web interactive permet d'explorer les données et de faire des prédictions.

**Fonctionnalités :**

* **📊 Onglet EDA** : graphiques Plotly interactifs (ventes par semaine, ventes par type de magasin, comparaison d'une année sur l'autre, effet des jours fériés, promotions et ventes)
* **📈 Onglet Model Performance** : WMAE, MAE et R² sur l'année de validation avec les fêtes, ventes réelles et prédites pour toute la chaîne et pour n'importe quel couple magasin/département, importance des variables
* **🔮 Onglet Predict** : on choisit un magasin, un département et **n'importe quelle date** (passée, ou future jusqu'en 2040). Les informations du magasin, les facteurs externes et les ventes passées sont remplis automatiquement. Après la fin des données, les facteurs externes sont estimés, les ventes passées viennent de la même semaine de l'année la plus récente disponible (alignée sur les fêtes : Thanksgiving 2016 est comparé à Thanksgiving 2011), et l'explication de l'IA précise que la prévision est moins sûre à mesure qu'on s'éloigne. Ensuite, la combinaison prédit les ventes de la semaine (avec le détail des parts LightGBM et naïve saisonnière), et un **assistant IA explique la prédiction** (voir ci-dessous)
* **💬 Onglet Ask the data** : on pose une question en anglais ; un agent IA écrit et exécute du SQL pour y répondre, en affichant chaque requête (voir ci-dessous)

**Lancer le projet :**

```bash
# Python 3.11+
pip install -r requirements.txt

# télécharger les données de la compétition dans Data/ (accepter d'abord le règlement de la compétition sur Kaggle)
kaggle competitions download -c walmart-recruiting-store-sales-forecasting
unzip walmart-recruiting-store-sales-forecasting.zip -d Data

# lancer l'application
streamlit run app.py
```

Ouvrez ensuite `http://localhost:8501` dans votre navigateur.

**Ou avec Docker** (l'image ne contient que l'application et le modèle entraîné ; le dossier local `Data/` est monté en lecture seule) :

```bash
# après avoir téléchargé les données dans Data/ (voir ci-dessus)
docker compose up -d --build     # construit l'image et lance l'application sur http://localhost:8501
docker compose down              # l'arrête
```

---

## 🤖 Explication de chaque prédiction par l'IA (RAG)

Après chaque prédiction, l'application **l'explique automatiquement en langage clair**, grâce au RAG, la génération augmentée par la recherche (`walmart_explain.py`) :

1. **Recherche** : les faits derrière la prédiction sont extraits des données : ventes de la même semaine l'an dernier, niveau habituel des ventes, prochaine fête et ventes de ce département cette semaine-là l'an dernier, promotions en cours, et erreur habituelle du modèle à cette période de l'année (d'après la validation).
2. **Augmentation** : ces faits sont placés dans un prompt qui interdit au LLM d'inventer ou de calculer un nombre.
3. **Génération** : un **LLM local servi par [Ollama](https://ollama.com)** rédige l'explication, affichée mot par mot. Pas de clé API, pas de coût.

Tous les chiffres viennent des données, jamais du LLM, et les faits utilisés sont affichés sous l'explication pour pouvoir les vérifier. **Sans Ollama, l'application fonctionne quand même** : une explication plus simple est construite à partir des mêmes faits, avec des modèles de phrases.

```bash
# optionnel : installer Ollama, puis télécharger un modèle
ollama pull qwen3.5:9b          # n'importe quel modèle convient ; le plus petit (le plus rapide) est utilisé par défaut
OLLAMA_MODEL=qwen3.5:27b streamlit run app.py   # optionnel : forcer un autre modèle
```

Avec Docker, le conteneur accède à Ollama qui tourne sur la machine hôte (`OLLAMA_HOST` dans `docker-compose.yml`).

### Vérification de fidélité

Après chaque explication, `verify_numbers()` extrait tous les nombres écrits par le LLM (en chiffres ou en lettres, par exemple « twenty-one ») et vérifie qu'ils viennent bien des faits récupérés, en tolérant les arrondis normaux. Si un chiffre n'est pas dans les faits, l'explication de l'IA est **remplacée par la version construite à partir des modèles de phrases** et les chiffres inventés sont listés ; sinon, l'application affiche « ✓ Every figure was checked against the facts ».

**Évaluation** (`evaluate_rag.py`, 200 prédictions tirées au hasard avec une graine fixe : 1/3 de semaines passées, 1/3 de la période de test Kaggle, 1/3 de semaines futures de 2014 à 2020) :

| Modèle | Explications sans chiffre inventé | Temps médian |
| --- | --- | --- |
| qwen3.5:9b | **200 / 200 (100 %)** | 5,4 s |

Résultats détaillés dans `evaluation/rag_evaluation_qwen3.5_9b.csv`. Limite : la vérification confirme qu'un nombre existe dans les faits, pas qu'il est utilisé au bon endroit. Dans un cas sur 200, « over five years ago » (un nombre calculé par le modèle lui-même) est passé uniquement parce que la semaine comptait aussi 5 promotions. Les erreurs d'interprétation sans nombre (par exemple « Thanksgiving peut faire monter les ventes » pour un département qui vend moins cette semaine-là) ne sont pas détectées non plus ; il faudrait pour cela un second LLM jouant le rôle de juge.

```bash
python evaluate_rag.py                     # 200 cas avec le plus petit modèle installé
python evaluate_rag.py --model qwen3.5:27b --n 50
python evaluate_rag.py --recheck           # revérifie les explications enregistrées sans rappeler le LLM
```

---

## 💬 Interroger les données : agent SQL (IA agentique avec LangGraph)

L'onglet **💬 Ask the data** répond à des questions ouvertes sur les données (« Quels sont les 5 magasins qui ont le plus vendu pendant les semaines de Thanksgiving ? ») grâce à un **agent construit avec LangGraph** qui écrit et exécute lui-même du SQL (`walmart_agent.py`) :

```
question -> [agent] --appel d'outil--> [tools] --résultat ou erreur--> [agent] --> ... --brouillon--> [review]
               ^                                                                                       |
               +--------------------------------- problèmes trouvés -----------------------------------+
                                                                                  approuvé --> réponse
```

* **agent** : le LLM local (Ollama) décide de la suite, avec 4 outils : `list_tables`, `describe_table`, `run_sql` et `make_chart`. Quand une requête échoue, il lit l'erreur et corrige sa requête.
* **graphiques** : quand on lui demande un graphique, l'agent appelle `make_chart` avec une requête `SELECT`, un type de graphique (courbe, barres, barres horizontales, nuage de points) et les colonnes. **C'est notre code qui dessine un graphique Plotly interactif** (info-bulle sur chaque barre ou point, repère vertical sur les courbes, zoom, export PNG), dans le thème clair ou sombre de l'application, avec une palette vérifiée pour le daltonisme : le LLM n'écrit jamais de code exécuté, et la requête passe par les mêmes contrôles de lecture seule. Les données du graphique sont renvoyées à l'agent, qui le décrit avec de vrais chiffres.
* **revue** : avant de répondre, des vérifications automatiques contrôlent que **chaque chiffre de la réponse apparaît dans un résultat de requête** (les arrondis comme « $483.9 million » sont acceptés, un chiffre calculé par le LLM lui-même doit être calculé en SQL) et qu'**un graphique demandé a bien été créé** ; puis le LLM relit la logique (grain des données, moyennes hebdomadaires, périodes alignées). Chaque réponse est revue au plus deux fois.
* **couche sémantique** : en plus des tables brutes, `store_weeks` (une ligne par magasin et par semaine, départements déjà additionnés) et `holiday_weeks` (le nom de chaque semaine de fête) évitent les erreurs les plus fréquentes d'un petit modèle.
* **sécurité** : les CSV sont chargés dans une **base DuckDB en mémoire, puis l'accès aux fichiers est coupé et la configuration verrouillée**. Une seule requête `SELECT` est acceptée (pas d'`INSERT`, `DELETE`, `DROP`, `COPY` ni `SET`), les résultats sont limités à 50 lignes, une requête répétée est refusée et le nombre d'étapes est plafonné.
* **transparence** : chaque requête SQL et son résultat sont affichés dans l'application, pour pouvoir vérifier la réponse.

**Évaluation** (`evaluate_agent.py`) : 10 questions dont la réponse attendue est recalculée par une requête SQL de référence à chaque exécution.

| Modèle | Réponses correctes | Temps médian |
| --- | --- | --- |
| qwen3.5:9b | **9 à 10 / 10** selon l'exécution | 5 à 6 s |

Limites : 10 questions forment un petit test, et le même agent obtient 9 ou 10 selon l'exécution : un modèle de 9B n'applique pas toujours ses consignes (le classement de Thanksgiving cite parfois deux fois le même magasin malgré une règle qui l'interdit), et la revue ne le détecte pas toujours. Les vérifications portent sur les chiffres, pas sur les dates ou les affirmations écrites en toutes lettres. Les questions ambiguës (par exemple la croissance entre deux années, quand les périodes peuvent être découpées de plusieurs façons) peuvent recevoir des réponses différentes mais valables. `qwen3.5:27b` a été testé : il n'était pas plus précis, et 10 à 20 fois plus lent.

```bash
python evaluate_agent.py --model qwen3.5:9b
```

---

## 💡 À retenir

* **Les modèles linéaires échouent complètement** (R² ≈ 0,09) : les données sont très non linéaires, à cause des pics des fêtes et des interactions entre magasins et départements
* **AdaBoost s'effondre** (R² = -0,47) : il est trop sensible aux ventes extrêmes des fêtes
* **Les modèles à base d'arbres dominent** : un seul arbre de décision atteint déjà R² = 0,90, et les ensembles d'arbres font mieux encore
* **Le travail sur les variables compte plus que le choix du modèle** : ajouter le niveau de ventes de chaque couple magasin/département et les décalages de 52 semaines réduit le WMAE de LightGBM de 41 %, bien plus que n'importe quel réglage ou changement d'algorithme
* **Une méthode simple peut battre un modèle complexe** : « même semaine l'an dernier » (débruitée entre magasins) bat LightGBM à elle seule, et la moyenne des deux donne le meilleur résultat
* **Les variables liées aux fêtes ont des résultats mitigés** : `Days_to_Thanksgiving` est utile, mais `Days_to_SuperBowl` et `Days_to_LaborDay` sont ignorées par le modèle
* **La métrique change le gagnant** : parmi les modèles de base, XGBoost est le meilleur en R², et Random Forest avec le WMAE officiel
* **La période de validation compte** : une validation sans Thanksgiving ni Noël donnait des scores trop optimistes
* **Le découpage temporel entre entraînement et validation** est indispensable pour des séries temporelles : un découpage aléatoire ferait fuiter des données futures et gonflerait les scores

---

## 🧪 Expériences (`Notebook_Experiments.ipynb`)

Trois améliorations testées séparément avant d'être intégrées au modèle final :

**1. Une validation qui contient les fêtes.** Nouveau découpage : entraînement jusqu'au 28/10/2011, validation sur les 52 semaines suivantes (04/11/2011 → 26/10/2012), qui **incluent Thanksgiving et Noël 2011**. Le WMAE de tous les modèles se dégrade (+6 % à +18 %) : les scores d'origine étaient trop optimistes.

**2. Variables de décalage** (seulement des décalages de 52 semaines ou plus, car le test de Kaggle va jusqu'à 39 semaines à l'avance) : ventes du même couple magasin/département 52 semaines plus tôt, leur moyenne sur 4 semaines, et la moyenne et la médiane historiques de chaque couple.

| Modèle (nouveau découpage) | WMAE de base | WMAE + décalages | Gain |
| --- | --- | --- | --- |
| **LightGBM** | 3 501 | **2 070** | **-41 %** |
| Random Forest | 2 779 | 2 293 | -17 % |
| XGBoost | 2 919 | 2 645 | -9 % |

**LightGBM avec les décalages est le meilleur modèle seul du projet** (WMAE 2 070, R² 0,9637), et le seul qui s'améliore aussi sur les semaines de Thanksgiving et Noël (MAE 4 605 → 2 989). L'essentiel du gain vient du niveau historique de chaque couple magasin/département.

**Encodage out-of-fold.** Calculées sur tout l'entraînement, `Store_Dept_Mean` et `Store_Dept_Median` laissaient chaque ligne voir sa propre vente. Avec un encodage out-of-fold (5 groupes de semaines), **le WMAE de XGBoost passe de 2 645 à 2 061 et son R² remonte de 0,812 à 0,965** : il exploitait cette fuite. LightGBM (+1 %) et Random Forest (+3 %) ne sont pas touchés, donc la partie LightGBM du modèle final garde l'encodage plus simple.

**Idées du gagnant de la compétition** (David Thaler, [code](https://github.com/davidthaler/Walmart_competition_code)), testées une par une contre LightGBM avec décalages (WMAE 2 068) :

| Idée | WMAE | Verdict |
| --- | --- | --- |
| Retirer les variables externes (CPI, MarkDowns, etc.) | 2 034 (-1,6 %) | petit gain, non conservé en plus de la combinaison |
| Correction de Noël selon le jour de la semaine | 2 098 à 2 222 | ❌ LightGBM capte déjà cet effet |
| Décalages calculés sur l'historique débruité par SVD | 2 071 à 2 079 | ➖ neutre pour LightGBM |
| **Moyenne avec une prévision naïve saisonnière SVD** | **1 822 (-12 %)** | ✅ **adoptée** |

**Validation sur 3 périodes.** Pour vérifier que ces gains ne viennent pas de la chance d'une seule période de validation, le modèle final et ses deux parties sont évalués sur 3 fenêtres de 39 semaines (l'horizon du test) :

| Modèle | Fenêtre A | Fenêtre B | Fenêtre C | Moyenne ± écart-type |
| --- | --- | --- | --- | --- |
| LightGBM + décalages | 2 171 | 2 183 | 1 707 | 2 020 ± 271 |
| Naïve saisonnière (SVD) | 1 941 | 1 960 | 1 738 | 1 880 ± 123 |
| **Moyenne 50/50** | **1 875** | **1 889** | **1 635** | **1 799 ± 143** |
| Moyenne 30/70 | 1 853 | 1 867 | 1 654 | 1 791 ± 119 |

La combinaison bat ses deux parties **dans chaque fenêtre**. Les scores varient d'environ 8 % d'une fenêtre à l'autre, donc des écarts de 1 % mesurés sur une seule période relèvent du bruit : la pondération 30/70 gagne sur deux fenêtres et perd sur la troisième, ce qui confirme le choix de garder le poids 50/50 fixé à l'avance.

**3. Réentraînement sur toutes les données (2010 → 2012)** avant de prédire le test. La prévision obtenue retrouve le niveau de la même saison un an plus tôt, alors que le modèle d'origine, qui n'avait jamais vu 2012, était environ 5 % en dessous.

---

# 📚 Détails

## 📁 Données

Données issues de la compétition Kaggle [Walmart Recruiting - Store Sales Forecasting](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting). **Les données ne sont pas incluses dans ce dépôt** : elles appartiennent à la compétition et doivent être téléchargées sur Kaggle après avoir accepté son règlement (voir *Lancer le projet* plus haut).

| Fichier                  | Description                                                                   |
| ------------------------ | ----------------------------------------------------------------------------- |
| `train.csv`              | Ventes hebdomadaires historiques par magasin/département                      |
| `test.csv`               | Magasins/départements et semaines à prédire                                   |
| `features.csv`           | Variables externes (température, prix du carburant, promotions, CPI, chômage) |
| `stores.csv`             | Type (A, B, C) et taille de chaque magasin                                    |
| `sampleSubmission.csv`   | Format de soumission                                                          |

**Jeu d'entraînement fusionné :** 421 570 lignes

---

## 📂 Structure du projet

```
shelfcast/
│
├── Data/                   # absent du dépôt : à télécharger sur Kaggle (voir Lancer le projet)
│   ├── train.csv
│   ├── test.csv
│   ├── features.csv
│   ├── stores.csv
│   └── sampleSubmission.csv
│
├── Notebook.ipynb          # exploration, variables, modèles, évaluation, soumission
├── Notebook_Experiments.ipynb  # validation avec les fêtes, décalages, réentraînement sur toutes les données
├── walmart_features.py     # calcul des variables, partagé par le notebook et l'application
├── walmart_explain.py      # explication RAG d'une prédiction (recherche + LLM local via Ollama + vérification de fidélité)
├── evaluate_rag.py         # mesure les chiffres inventés dans les explications sur 200 prédictions au hasard
├── walmart_agent.py        # agent SQL construit avec LangGraph (base DuckDB en lecture seule, étape de revue)
├── walmart_charts.py       # graphiques Plotly interactifs dessinés pour l'agent (thème clair ou sombre, palette validée)
├── evaluate_agent.py       # mesure la précision de l'agent sur des questions à réponse SQL de référence
├── evaluation/             # résultats des évaluations (explications RAG et agent SQL)
├── tests/                  # tests unitaires de walmart_features.py (python -m pytest)
├── app.py                  # application Streamlit (exploration, performance, prédiction)
├── walmart_lgbm_model.pkl  # partie LightGBM du modèle final, entraînée sur toutes les données (2,9 Mo)
├── model_validation.csv.gz # prédictions de validation affichées dans l'application
├── requirements.txt        # tout (application + notebooks + tests)
├── requirements-app.txt    # application seule, utilisé par l'image Docker
├── Dockerfile
├── .github/workflows/ci.yml  # CI : tests + build Docker à chaque push
├── docker-compose.yml
├── pytest.ini
├── README.md               # version anglaise
└── README.fr.md            # version française
```

---

## 🔍 Analyse exploratoire

Principaux enseignements :

* **Les ventes culminent** autour des semaines 7, 14, 22, 27, 40, 47 et 51, surtout à cause des fêtes
* **Les magasins de type A** dominent le volume de ventes, suivis des types B et C
* **Les magasins de type C** ont des ventes stables toute l'année, alors que les types A et B ont de forts pics saisonniers
* **Les semaines fériées** ont des ventes moyennes environ 7,1 % plus élevées, alors qu'elles ne représentent que 10 semaines sur 143
* **Les promotions (MarkDowns) suivent les pics de ventes**, mais les plus grosses promotions ont lieu pendant les fêtes : une partie de ce lien vient donc du calendrier lui-même
* **La taille du magasin** (0,24) est la variable la plus corrélée aux ventes
* Les totaux annuels donnent l'impression d'une forte baisse en 2012, mais c'est un **effet de couverture des données** : elles vont du 05/02/2010 au 26/10/2012, donc 2012 n'a pas son pic de novembre et décembre. Sur les **39 semaines présentes dans les trois années**, les ventes sont en réalité **stables** (1 797 M$ → 1 786 M$ → 1 831 M$)

---

## ⚙️ Variables créées

| Variable                 | Description                            |
| ------------------------ | -------------------------------------- |
| `Year`                   | Extraite de `Date`                     |
| `Week_Of_Year`           | Numéro de la semaine (1 à 52)          |
| `Days_to_Thanksgiving`   | Jours avant ou après Thanksgiving      |
| `Days_to_Christmas`      | Jours avant ou après Noël              |
| `Days_to_SuperBowl`      | Jours avant ou après le Super Bowl     |
| `Days_to_LaborDay`       | Jours avant ou après Labor Day         |

Le mois et le jour du mois ne sont pas créés, car ils feraient doublon avec `Week_Of_Year`. `Date` est retirée une fois ces variables extraites.

---

## 🧹 Prétraitement

```python
# Encodage ordinal du type de magasin (C=0 < B=1 < A=2, dans l'ordre du volume de ventes)
OrdinalEncoder(categories=[["C", "B", "A"]])

# Standardisation de toutes les variables numériques
StandardScaler()

# Le tout assemblé avec ColumnTransformer + Pipeline
ColumnTransformer([
    ("Categorical", cat_pipeline, cat_data),
    ("Numerical",   num_pipeline, num_data)
], remainder="drop")
```

---

## 📊 Découpage entraînement / validation

Découpage temporel pour éviter les fuites de données : on entraîne sur le passé et on valide sur le futur.

| Jeu        | Années      | Lignes  |
| ---------- | ----------- | ------- |
| Entraînement | 2010 et 2011 | 294 132 |
| Validation | 2012        | 127 438 |

---

## 🤖 Modèles et résultats

### Métrique officielle de Kaggle : le WMAE

La compétition est notée avec le **Weighted Mean Absolute Error** (erreur absolue moyenne pondérée), où les semaines fériées pèsent **5 fois** plus que les semaines normales :

$$WMAE = \frac{\sum_i w_i \, |y_i - \hat{y}_i|}{\sum_i w_i}, \qquad w_i = 5 \text{ si semaine fériée, sinon } 1$$

| Modèle                       | **WMAE**      | MAE    | RMSE      | R² validation  | R² entraînement | Écart  |
| ---------------------------- | ------------- | ------ | --------- | -------------- | --------------- | ------ |
| **Random Forest (par défaut)** | **2 286** 🥇 | 2 273  | 5 003     | 0,9488         | 0,9962          | 0,0474 |
| Random Forest (réglé)        | 2 367         | 2 333  | 5 149     | 0,9458         | 0,9728          | 0,0270 |
| **XGBoost (réglé)**          | 2 752         | 2 732  | **4 829** | **0,9523** 🥇  | 0,9759          | 0,0236 |
| Arbre de décision (par défaut) | 2 988       | 2 968  | 6 717     | 0,9078         | 1,0000          | 0,0922 |
| LightGBM                     | 3 149         | 3 122  | 5 324     | 0,9421         | 0,9553          | 0,0132 |
| XGBoost (par défaut)         | 3 361         | 3 318  | 5 491     | 0,9384         | 0,9470          | 0,0086 |
| Arbre de décision (réglé)    | 4 664         | 4 624  | 8 132     | 0,8648         | 0,8656          | 0,0007 |
| SGD Regressor                | 14 155        | 14 145 | 21 326    | 0,0705         | 0,0828          | n/a    |
| Régression linéaire          | 14 374        | 14 299 | 21 118    | 0,0886         | 0,0854          | n/a    |
| AdaBoost                     | 22 599        | 22 727 | 26 875    | -0,4761        | -0,6408         | n/a    |

*Écart = R² d'entraînement moins R² de validation. Un écart important signifie que le modèle mémorise les données d'entraînement. Il n'a pas de sens pour les modèles qui échouent sur les deux jeux.*

**Le classement dépend de la métrique.** XGBoost gagne en R² et en RMSE, qui élèvent les erreurs au carré et le récompensent de mieux gérer les quelques ventes très élevées. **Random Forest gagne en WMAE et en MAE**, les métriques qui reflètent l'erreur sur une semaine typique, dont celle qu'utilise Kaggle. Le Random Forest par défaut a le plus petit WMAE, mais aussi le plus grand écart de surapprentissage (0,047) : c'est donc la version réglée qu'il est plus sûr de comparer à XGBoost ci-dessous.

> ⚠️ **Limite de cette validation :** la validation 2012 s'arrête le 26/10/2012, donc elle ne contient **ni Thanksgiving ni Noël**. Ses seules semaines fériées sont le Super Bowl et Labor Day. Ces scores sont donc optimistes par rapport au test de Kaggle (novembre 2012 → juillet 2013), qui contient les deux pics.

---

## ⚖️ Choix parmi les modèles de base

**Au stade des modèles de base (variables ci-dessus, validation 2012) :** aucun modèle ne gagne partout.

| | Random Forest (réglé) | XGBoost (réglé) |
| --- | --- | --- |
| WMAE (métrique Kaggle) | **2 367** | 2 752 (+16 %) |
| R² / RMSE | 0,9458 / 5 149 | **0,9523 / 4 829** |
| Taille du modèle sauvegardé | 182 Mo (64 Mo compressé) | **3,9 Mo** |

Random Forest gagne sur la métrique notée, mais il est trop lourd pour être déployé confortablement (limite de 100 Mo par fichier sur GitHub, chargement lent sur Streamlit Cloud). Aucun des deux n'est le modèle final : les expériences présentées plus haut montrent que **LightGBM avec les décalages**, combiné à une prévision naïve saisonnière, bat les deux.

---

## 📈 Prédictions face à la réalité et explicabilité (XGBoost de référence)

Sur 2012 (jamais vu à l'entraînement), le **total des ventes hebdomadaires** prédit par XGBoost pour l'ensemble des magasins s'écarte en moyenne de **2,59 %** de la réalité. Le notebook compare les ventes prédites et réelles pour toute la chaîne et pour des couples magasin/département.

**L'importance des variables et SHAP** montrent que :

* **Dept, Size, Store et Type** expliquent l'essentiel des prédictions : *ce qui* est vendu et *où* comptent bien plus que tout le reste.
* **Le CPI** a un effet négatif, mais il est régional et presque constant pour un magasin donné. Le modèle s'en sert donc très probablement pour **reconnaître la localisation du magasin**, pas comme un facteur économique.
* La saisonnalité vient de `Week_Of_Year` et `Days_to_Thanksgiving`. `Days_to_SuperBowl` et `Days_to_LaborDay` ont une **importance nulle** et pourraient être retirées.
* Les promotions, la température et le prix du carburant ont un faible impact.

---

## ✅ Tests

`tests/test_walmart_features.py` vérifie les fonctions qui échoueraient **sans bruit** si elles étaient fausses (un mauvais score, pas un plantage) :

* le décalage de 52 semaines prend exactement la même semaine un an plus tôt (Thanksgiving 2011 ↔ Thanksgiving 2010),
* les décalages et les statistiques par magasin/département ne lisent que l'historique fourni (**aucune fuite de données futures**),
* la distance aux fêtes vaut 0 la semaine fériée, et les promotions manquantes deviennent 0,
* la SVD avec toutes ses composantes redonne les ventes d'origine et n'invente jamais de semaines non observées,
* la combinaison vaut exactement 50 % LightGBM + 50 % naïve saisonnière, et le WMAE donne bien un poids de 5 aux semaines fériées,
* le RAG récupère les bons faits (l'an dernier, la prochaine fête, les promotions en cours, l'erreur passée à la même période), met chaque chiffre dans le prompt, retire le raisonnement caché du modèle (`<think>`) et transmet la réponse mot par mot depuis un **faux serveur Ollama** lancé dans les tests.

```bash
python -m pytest     # 74 tests, environ 10 secondes, sans les données Kaggle ni LLM
```

Décaler le décalage d'une seule semaine fait échouer 5 de ces tests.

**Intégration continue :** une GitHub Action (`.github/workflows/ci.yml`) lance ces tests à chaque push, construit l'image Docker et vérifie que l'application démarre.

---

## 📤 Soumission Kaggle

La dernière section de `Notebook.ipynb` génère `walmart_submission.csv` avec le modèle final, au format attendu par Kaggle (`Id,Weekly_Sales`, 115 064 lignes), avec une vérification par rapport à `sampleSubmission.csv`.

```bash
kaggle competitions submit -c walmart-recruiting-store-sales-forecasting -f walmart_submission.csv -m "LightGBM + seasonal naive ensemble"
```

| Soumission | WMAE public | WMAE privé |
| --- | --- | --- |
| LightGBM + décalages | 2 929 | 3 070 |
| **Modèle final (LightGBM + naïve saisonnière SVD)** | **2 769** | **2 885** |

La combinaison améliore le score réel de 6 %, ce qui confirme le sens du gain observé sur la validation locale (-12 %).

---

## 🛠️ Technologies

| Bibliothèque         | Usage                                          |
| -------------------- | ---------------------------------------------- |
| `pandas`, `numpy`    | Manipulation des données                       |
| `scikit-learn`       | Prétraitement, pipelines, modèles              |
| `lightgbm`           | Modèle final (partie de la combinaison)        |
| `xgboost`            | Modèle de référence                            |
| `shap`               | Explicabilité du modèle                        |
| `plotly`             | Graphiques interactifs                         |
| `streamlit`          | Application web                                |
| `langgraph`, `langchain-ollama` | Agent SQL (IA agentique) |
| `duckdb`             | Base SQL en mémoire, en lecture seule, pour l'agent |
| `matplotlib`         | Graphiques SHAP dans les notebooks |
| Docker               | Application conteneurisée                      |
| `joblib`             | Sauvegarde du modèle                           |
| CUDA (RTX 5070)      | Accélération GPU pour la comparaison des modèles |
