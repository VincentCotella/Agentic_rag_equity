# README - 10-K Report Generator & Financial Data Analysis

Bienvenue dans ce projet combinant **RAG (Retrieval-Augmented Generation)** pour l’analyse de rapports 10-K et un **module** d’analyse de données financières (CSV). Cette application Streamlit vous permet :

1. De **générer** un rapport en plusieurs parties (Part I à Part V) sur un 10-K (par ex. Apple, Microsoft), soit via **RAG** (vector DB) soit via des **sections brutes**.
2. D’**analyser** un fichier CSV (colonnes financières, etc.), de le filtrer et d’en **générer** des graphiques interactifs via **Plotly** (contrôlé par un agent LLM).


## Table des Matières

1. [Fonctionnalités Clés](#fonctionnalités-clés)  
2. [Installation & Configuration](#installation--configuration)  
   - [Configuration du fichier `.env`](#configuration-du-fichier-env)
3. [Arborescence](#arborescence)  
4. [Usage](#usage)  
   - [Démarrer l’application](#démarrer-lapplication)  
   - [Analyse 10-K (RAG ou Raw)](#analyse-10-k-rag-ou-raw)  
   - [Analyse CSV & Graphiques](#analyse-csv--graphiques)  
5. [Configuration LLM & Groq](#configuration-llm--groq)  
6. [Remarques sur le Parsing 10-K](#remarques-sur-le-parsing-10-k)  


## Fonctionnalités Clés

1. **Génération de Rapport 10-K**  
   - Découpage du rapport en 5 parties : Part I (présentation générale), Part II (chiffres financiers), Part III (performance annuelle), Part IV (positionnement marché), Part V (risques).  
   - Choix entre :
     - **RAG (vector DB)** : interroge une base Chroma (documents vectorisés).  
     - **Raw 10-K** : récupère directement les sections (Item 1, 1A, etc.) via la librairie `edgar`.

2. **Analyse Financière de CSV**  
   - Chargement d’un CSV.  
   - Filtrage de colonnes.  
   - Création de graphiques Plotly (ou Plotly Express) en fonction d’une **requête utilisateur** guidée par un **agent LLM**.  
   - Approches “données récentes” ou “historique” via différents prompts.

3. **Gestion de la base Chroma**  
   - Réinitialisation (Reset).  
   - Ajout de documents `.txt` ou `.md` pour l’indexation vectorielle (RAG).  
   - Visualisation des chunks / documents indexés.

---

## Installation & Configuration

1. **Cloner** ce dépôt :
   ```bash
   git clone <URL-du-repo>
   cd rag_equity
   ```

2. **Environnement Python** :
   - Créer un venv ou un conda env :
     ```bash
     python -m venv myenv
     source myenv/bin/activate  # Linux/Mac
     # ou myenv\Scripts\activate  # Windows
     ```
3. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```
   Assurez-vous que les librairies suivantes (et leurs versions) sont présentes :
   - `plotly`
   - `langchain` / `langchain-experimental`
   - `langchain-groq`
   - `edgar` (pour parsing 10-K)
   - `pandas`, etc.

### Configuration du fichier `.env`
Vous devez créer un fichier `.env` à la racine du projet contenant, au minimum :

```env
MODEL_NAME=llama3-70b-8192
GROQ_API_KEY=<votre_cle_GROQ>
```

- **MODEL_NAME** : Nom du modèle LLM à utiliser via l’API GROQ (ex: `"llama3-70b-8192"`).  
- **GROQ_API_KEY** : votre clé d’API pour GROQ.  

D’autres variables peuvent être ajoutées si nécessaire (ex. `OPENAI_API_KEY` si vous utilisez OpenAI).

---

## Arborescence

rag_equity/
├─ app.py                     # Application Streamlit principale
├─ requirements.txt           # Liste des dépendances
├─ config.py                  # Variables globales (chemins, clés)
├─ core/
│  ├─ edgar_direct_manager.py # Gère la récupération du 10-K
│  ├─ data_management.py      # Gestion de la base Chroma (clear, add docs, etc.)
│  ├─ groq_llm.py             # Classe GROQLLM (LangChain)
│  ├─ multi_agentic_rag.py    # Coordonne les 5 agents (Part I..V)
│  └─ ...
├─ agents/
│  ├─ base_agent.py           # Classe mère Agent
│  ├─ unstructured_data_agent.py # RAG agent -> Chroma
│  ├─ report_part1_agent.py   # Gère Part I
│  ├─ report_part2_agent.py   # Gère Part II
│  ├─ report_part3_agent.py
│  ├─ report_part4_agent.py
│  └─ report_part5_agent.py
├─ temp_uploads/             # Fichiers uploadés (pour Chroma ou CSV)
├─ .env                       # Contient MODEL_NAME, GROQ_API_KEY, etc.
└─ ...

---

## Usage

### Démarrer l’application

1. Activez votre venv :
   ```bash
   source myenv/bin/activate  # ou Windows: myenv\Scripts\activate
   ```
2. Lancez Streamlit :
   ```bash
   streamlit run app.py
   ```
3. Ouvrez le lien local `http://localhost:8501` dans votre navigateur.

### Analyse 10-K (RAG ou Raw)

- **Dans la section “Generate the 10-K Report”** :  
  1. Entrez un **ticker symbol** (ex: `AAPL`).  
  2. Cliquez sur “Initialize System” pour créer l’objet `MultiAgenticRAG`.  
  3. Choisissez “Use RAG (vector DB)” ou “Use raw 10-K sections”.  
  4. Cliquez sur “Generate Part I” (ou II, III, etc.) pour générer le rapport correspondant.  

Vous pouvez **téléverser** des documents `.txt / .md` vers Chroma afin d’alimenter la base vectorielle (RAG).

### Analyse CSV & Graphiques

1. Dans la section **Financial CSV Data Analysis** :  
   - **Uploader** un fichier CSV.  
   - Affichez un **aperçu** du DataFrame, sélectionnez les **colonnes**.  
   - Entrez une **requête** (ex.: “Plot the average revenue by region”).  
   - Un **agent** LLM générera le code Plotly, puis l’exécutera pour afficher le graphique.


## Configuration LLM & Groq

- Par défaut, on utilise un LLM fourni par **GROQ** (via `groq` / `langchain_groq`).  
- Les variables d’environnement `MODEL_NAME` et `GROQ_API_KEY` (voir `.env`) doivent être configurées.  
- Si vous préférez OpenAI ou un autre service, adaptez la classe `groq_llm.py` ou le code `llm = ChatGroq(...)` dans `app.py`.


## Remarques sur le Parsing 10-K

- Le code se base sur `edgar` pour récupérer les sections (`Item 1`, etc.).  
- Selon la version, `.obj()` peut renvoyer un **dict** ou un objet `TenK`.  
- Nous avons un `EdgarDirectManager` qui récupère le texte de différents Items, parfois en le concaténant.  
- Attention aux **limites de tokens** si vous concaténez trop de sections à la fois.

