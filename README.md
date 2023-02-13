# Cancelled-Reservation-Predictor

Progetto di intelligenza artificiale il quale obiettivo è prevedere se una prenotazione per una stanza di albergo sarà rispettata o meno

Componenti del team:

| Nome        | Cognome         | Matricola|
| ------------- |:-------------:| --------:|
| Cristian  | Vacchiano| 0512105910 |
| Luigi  | Ciriello    |   0512105868 |

# Informazioni riguardo il progetto

L'intero progetto rispetta in gran parte il modello CRISP-DM pertanto è stato modularizzato nelle seguenti fasi:
- Business understanding: questa fase, che prevede l'esplorazione e la conoscenza del dominio, è stata eseguita direttamente nel primo paragrafo della documentazione.
- Data understanding: per la ricerca, il caricamento, l'analisi e l'esplorazione dei dati è stata creata una sezione apposita nel progetto, sotto "src/model_creation/steps/data_understanding", in cui è stata creato un file con un metodo apposito eseguibile da "main.py" commentando tutte le righe tranne lo statement "du.data_understanding()" che effettua unicamente tutta la fase di data understanding.
- Data preparation: anche la data preparation, che consiste di data cleaning, feature scaling, feature selection e data balancing è stata effettuata in un file con dei metodi appositi, sotto "src/model_creation/steps/data_preparation". Anche quest'ultima è indipendente dal resto del progetto e quindi può essere effettuata in singolo come la fase precedente. Ognuna delle 4 fasi del data preparation è stata effettuata in un metodo apposito, ognuno richiamato poi nel metodo principale "data_preparation()".
- Data Modeling ed Evaluation: le ultime due fasi sono state accorpate in un unico file sotto "src/model_creation/steps/data_modeling_evaluation" avente il metodo "data_modeling_evaluation()", prelevando il dataset in output dalla fase precedente chiamato "final_dataset.csv" (contenuto nel path "src/model_creation/dataset/final_dataset.csv"). Il modello viene allenato e viene valutato sulle metriche utili agli algoritmi di classificazione. Inoltre il modello allenato viene salvato sotto "src/classifier/modello_finale.sav".
- Deployment: il deployment viene effettuato tramite l'utilizzo di un file che espone il servizio descritto nella documentazione, il quale elabora la decisione e la ritorna sotto forma di oggetto JSON. La parte FE è implementata nei file sotto la cartella "CRP_FE" che rappresenta una piccola pagina web che tramite il JS richiama il servizio esposto ed usa i dati di ritorno per visualizzare la risposta nell'HTML. Anche questa fase può essere eseguita indipendentemente dalle altre, a patto che esistano i 3 file prelevati da "src/classifier" (generati durante la fase precedente). Dopo aver avviato il server (api.py), può essere visualizzata la pagina "index.html" contenuta nella cartella "CRP_FE". Per la visualizzazione di quest'ultima è necessario avviare il file tramite live server di VS Code; in caso contrario potrebbero apparire dei problemi di CORS. La pagina avrà 200 prenotazioni create appositamente: il js darà in input al servizio ognuna di queste ed una volta ricevuto il JSON lo elaborerà e nella riga corrispondente alla prenotazione uscirà "Yes" se la prenotazione sarà cancellata, "No" altrimenti.

Saranno disponibili anche dei log per ognuna delle fasi in "src/logs" dove verranno mostrati i processi più importanti per ognuna di esse.

Le immagini generate nel progetto ed usate nella documentazione sono disponibili in "src/images"

Il progetto è stato sviluppato utilizzando l'IDE PyCharm.
