# Progetto Reddit 

## 📊 Descrizione
Progetto di analisi dei posts su Reddit, tramite diverse tecniche di machine learning sono classificati diversi posts in virali/non virali secondo vari criteri ricavati nello sviluppo del progetto.

## 🚀 Funzionalità
- **Data Collection**: Download dei posts tramite API ufficiale di reddit
- **Data Cleaning**: Trasformazione dei dati grezzi in features pronte per le fasi di addestramento
- **Unsupervised Learning**: Clustering 
- **Supervised Learning**: Classificazione secondo criteri dei clusters
- **Bayesian Networks**: Reti bayesiane (manuali) e Reti bayesiane con Hill Climbing

## ⬇️ Download repository

```bash
# Clona il repository
git clone https://github.com/Nikitaghq/Progetto_Reddit.git

# Installa le dipendenze
pip install -r requirements.txt
```
## 📦 Installazione
Non avviare file **reddit_scanner.py** senza aver letto la documentazione.

Il progetto dispone già dei dati necessari per l'avvio del programma e sono quelli discussi nella documentazione. 

Per testare il progetto con i dati già scaricati:
```bash
# Dopo aver installato le dipendenze
python main.py
```

Dopo aver letto la documentazione quindi si dispone del **username** e **password** fornita dall'**API di Reddit** allora seguire i seguenti passaggi:

1. Aprire il file **reddit_scanner** e inserire i dati in *client_id* e *client_secret* (riga 10-11)

2. Eseguire i seguenti comando da console:
```bash
# Scarica top 5000 posts su reddit
python reddit_scanner.py

# Pulisce i dati scaricati
python data_cleaner.py

# Solo dopo avvaire il main
python main.py
```
⚠️ **Attenzione il download dei posts impiega diverse ore (circa 3 ore)** ️️ ️
️️

⚠️ **Scaricando il nuovo dataset i risultati potrebbero differire da quelli discussi nel progetto**