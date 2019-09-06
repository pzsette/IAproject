# Progetto Intelligenza Artificiale

## Introduzione

Questo progetto prevede il testing dell’algoritmo di classificazione tramite alberi di decisone. 
È anche implementata una tecnica per la gestione di eventuali dati mancanti. 
Il test è effettuato più volte al variare della probabilità di trovare un valore mancante all'interno del dataset esaminato.

## Prerequisiti

- [ Python](https://www.python.org/) 3.7

- [matplotlib](https://matplotlib.org) Libreria utilizzata per la realizzazione di un grafico riassuntivo.

## Utilizzo del codice

Per riprodurre i risultati sottomessi è necessario eseguire il file test.py che esegue il test sui 3 datatet selezionati dal sito UCI ML. 
Ogni test è ripetuto quattro volte variando la probabilità di rimozione dei dati all’interno del dataset (0, 0.1, 0.2, 0.5). 
Ogni volta i dataset vengono testati con la tecnica _10-fold cross-validation_. Al termine dell'esecuzione è generato un grafico riassuntivo dei test

## Riferimenti

Per svolgere il progetto sono state utilizzate le seguenti risorse:
- [Artificial Intelligence - A Modern Approach](http://aima.cs.berkeley.edu)
- [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/) Utilizzato per ottenere i dataset su cui sono stati eseguiti i test.
- [aima-python repository](https://github.com/aimacode/aima-python) Fornisce il codice python per gli alberi di decisione e la tecnica _10-fold cross-validation_.
- Mitchell(1997) sezione 3.7.4 Sono riportate alcune tecniche per la gestione di dati mancanti all’interno di un albero di decisione tra cui quella scelta per questo progetto.
