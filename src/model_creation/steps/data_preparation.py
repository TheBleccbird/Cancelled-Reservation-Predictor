import pandas as pd
import sklearn
import steps.utils as utils
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

file = open("C:/Users/crist/PycharmProjects/Cancelled-Reservation-Predictor/src/logs/data_preparation.txt", "a")


def data_preparation():
    dataset = pd.read_csv("C:/Users/crist/PycharmProjects/Cancelled-Reservation-Predictor/src/model_creation/dataset/hotel_bookings.csv", sep=";")

    file.write("[FASE DI DATA PREPARATION]\n\n")

    # faccio data cleaning sul dataset
    cleaned_dataset = data_cleaning(dataset)

    # effettuo il feature scaling
    scaled_dataset = feature_scaling(cleaned_dataset)

    # codifico le variabili categoriche in numeriche
    no_cat_dataset = cat_to_num(scaled_dataset)

    # prima di effettuare la feature selection creo uno nuova colonna che conterrà sia le notti settimanali che
    # quelle del weekend
    no_cat_dataset = no_cat_dataset.assign(stays_nights=0)
    no_cat_dataset['stays_nights'] = no_cat_dataset['stays_in_weekend_nights'] + no_cat_dataset['stays_in_week_nights']
    no_cat_dataset = no_cat_dataset.drop(columns=['stays_in_weekend_nights', 'stays_in_week_nights'], axis=1)

    # nella feature selection è da escludere sicuramente is_repeated_guest per l'analisi fatta
    no_cat_dataset = no_cat_dataset.drop(columns=['is_repeated_guest'], axis=1)

    file.write("\n[FASE 3] DATA BALANCING\n\n")
    dataset_balanced = data_balancing(no_cat_dataset)

    #evaluate_classifiers(dataset_balanced)

    # effettuo la fase di feature selection
    file.write("\n[FASE 4] FEATURE SELECTION\n\n")
    selected_dataset = feature_selection(dataset_balanced, 18)

    final_dataset_creation(selected_dataset, list(selected_dataset), "final_dataset")

    return selected_dataset


def evaluate_classifiers(dataset):
    accuracys = [find_best_k_features(dataset, RandomForestClassifier()),
                 find_best_k_features(dataset, MultinomialNB()),
                 find_best_k_features(dataset, DecisionTreeClassifier()),
                 find_best_k_features(dataset, KNeighborsClassifier()),
                 find_best_k_features(dataset, LogisticRegression())]

    utils.create_evaluation_plot(accuracys, "Accuracy score")


def data_cleaning(dataset):
    file.write("[FASE 1] DATA CLEANING\n\n")

    # elimino i duplicati
    dataset = dataset.drop_duplicates()

    # elimino le feature "company" e "agent" dato che producono molti valori null
    dataset = dataset.drop(columns=["company", "agent"], axis=1)

    # elimino la feature "reservation_status_date" e "reservation_status" dato che serve poco al nostro scopo
    dataset = dataset.drop(columns=["reservation_status_date", "reservation_status"], axis=1)

    # elimino la feature "arrival_date_year" dato che si riferisce solo a 3 anni e quindi non gioverebbe a
    # prenotazioni effettuate non in quei 3 anni
    dataset = dataset.drop(columns=["arrival_date_year"], axis=1)

    # elimino anche la feature "deposit_type" dato che ha dei valori obsoleti
    dataset = dataset.drop(columns=["deposit_type"], axis=1)

    # elimino la riga che contiene adr negativo
    dataset = dataset.drop(dataset[(dataset["adr"] < 0)].index)

    # elimino gli outlier per la feature "adr"
    lower_bound, upper_bound = utils.detect_outliers(dataset, "adr")
    dataset.drop(dataset[(dataset["adr"] <= lower_bound) | (dataset["adr"] >= upper_bound)].index, inplace=True)

    # applico il most frequent imputation ai valori della feature "country" null
    dataset["country"] = dataset["country"].fillna(dataset['country'].value_counts().index[0])

    # applico il most frequent imputation ai valori della feature "children" null
    dataset["children"] = dataset["children"].fillna(dataset['children'].value_counts().index[0])

    # elimino le righe con 0 adulti, 0 bambini e 0 neonati
    dataset.drop(dataset[(dataset["adults"] == 0) & (dataset["children"] == 0) & (dataset["babies"] == 0)].index,
                 inplace=True)

    # elimino le righe con 0 adulti, almeno un bambino o almeno un neonato
    dataset.drop(dataset[(dataset["adults"] == 0) & ((dataset["children"] >= 0) | (dataset["babies"] >= 0))].index,
                 inplace=True)

    # SC e undefined nella feature meal riguardano la stessa cosa, sostituzione di tutti gli undefined -> SC
    dataset["meal"].replace("Undefined", "SC", inplace=True)

    # gli undefined in market segment e distribution channel sono anomali e li si imputano con la MFI
    mf_value = dataset["market_segment"].value_counts().index[0]
    dataset["market_segment"].replace("Undefined", mf_value, inplace=True)
    mf_value = dataset["distribution_channel"].value_counts().index[0]
    dataset["distribution_channel"].replace("Undefined", mf_value, inplace=True)

    utils.create_pie_chart(dataset, "Il bilanciamento del dataset dopo il data cleaning è:", "bilanciamento_rimozione_duplicati")
    balanced_count = dataset["is_canceled"].value_counts()
    file.write("\nIl bilanciamento del dataset dopo il data cleaning è: \n" + str(balanced_count) + "\n")

    return dataset


def feature_scaling(dataset):

    file.write("\n[FASE 2] FEATURE SCALING\n\n")
    # feature numeriche
    fields = ['lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'babies',
              'children', 'previous_cancellations', 'previous_bookings_not_canceled', 'booking_changes',
              'days_in_waiting_list', 'adr', 'total_of_special_requests', 'required_car_parking_spaces',
              'arrival_date_week_number', 'arrival_date_day_of_month']

    # varianza tra le feature numeriche
    file.write('\nLa varianza tra le feature numeriche è:\n' + str(dataset[fields].var()) + '\n')

    # quelli che hanno bisogno di scaling sono lead_time, days_in_waiting_list, adr, arrival_date_week_number e
    # arrival_date_day_of_month
    filter = ['lead_time', 'days_in_waiting_list', 'adr', 'arrival_date_week_number', 'arrival_date_day_of_month']

    # andiamo a scalare le variabili numeriche con il metodo del MinMax
    scaler = MinMaxScaler()

    dataset[filter] = scaler.fit_transform(dataset[filter])
    # salvo lo scaler per utilizzarlo successivamente
    utils.save_obj(scaler, 'src/classifier/scaler.sav')

    # varianza dopo aver applicato lo scaling
    file.write('\nLa varianza dopo aver applicato lo scaling tra le feature numeriche è:\n' + str(dataset[fields].var()) + '\n')

    return dataset


def cat_to_num(dataset):
    new_dataset, le = utils.convert_categorical(dataset, None, False)
    utils.save_obj(le, "src\classifier\encoder.sav")
    return new_dataset


def feature_selection(dataset, n):
    # divido il dataset in feature e target
    X = dataset.drop(columns=["is_canceled"])
    y = dataset["is_canceled"]

    selector = SelectKBest(chi2, k=n)
    selector.fit_transform(X, y)
    cols = selector.get_support(indices=True)

    X = X.iloc[:, cols]
    selected_features = list(X)
    if n == 18:
        file.write("\nLe feature selezionate sono:\n")
        file.write(str(n) + ")\n" + str(selected_features))

    print("\nLe feature selezionate sono:\n")
    print(str(n) + ")\n" + str(selected_features))
    selected_features.append("is_canceled")

    return dataset[selected_features]


def find_best_k_features(dataset, classifier):
    accurracy_list = []

    for n in range(2, len(dataset.columns)):
        dataset_selected = feature_selection(dataset, n)
        accuracy = classifier_accuracy(dataset_selected, classifier)

        accurracy_list.append(round(accuracy, 2)*100)

    return accurracy_list


def classifier_accuracy(dataset, classifier):
    without_target = dataset.drop(columns=["is_canceled"])
    X_train, X_test, y_train, y_test = train_test_split(without_target, dataset["is_canceled"], test_size=0.20,
                                                        random_state=42)
    classifier.fit(X_train, y_train)
    classifier_prediction = classifier.predict(X_test)

    return accuracy_score(y_test, classifier_prediction)


def data_balancing(dataset):
    # applicheremo la tecnica dell'undersampling
    # prelevo dal dataset di partenza tutte le istanze la cui prenotazione è cancellata
    minor_class_data = dataset[dataset["is_canceled"] == 1]

    # prelevo dal dataset di partenza 20000 istanze la cui prenotazione è rispettata
    major_class_data = dataset[dataset["is_canceled"] == 0].sample(n=20000, random_state=16)

    # unisco i due dataset creati in un unico dataset ed effettuo lo shuffle dei dati
    frames = [minor_class_data, major_class_data]
    dataset = pd.concat(frames)
    dataset = sklearn.utils.shuffle(dataset)

    count = dataset["is_canceled"].value_counts()
    file.write("\n\nIl bilanciamento nel dataset bilanciato è: \n" + str(count) + '\n')

    utils.create_pie_chart(dataset, "Il bilanciamento del dataset dopo il data balancing è: ","bilanciamento_data_balancing")

    return dataset


def final_dataset_creation(dataset, columns, file_name):
    # creo il dataset target
    final_dataset = pd.DataFrame(columns=columns)

    print("Inizio il salvataggio del nuovo csv")

    # inserisco tutti i nuovi dati nel file finale
    for i in range(len(dataset)):
        row = []
        print("Generazione della riga: " + str(i) + "/" + str(len(dataset)))
        for col in columns:
            column_value = dataset.iloc[i][col]
            row.append(column_value)
        final_dataset.loc[len(final_dataset)] = row

    # converto in csv il DataFrame
    final_dataset.to_csv("src/model_creation/dataset/" + file_name + ".csv", index=False)
