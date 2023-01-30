import pandas as pd
import steps.utils as utils

file = open("src/logs/data_preparation.txt", "a")


def data_preparation():
    dataset = pd.read_csv("src/model_creation/dataset/hotel_bookings.csv", sep=";")

    file.write("[FASE DI DATA PREPARATION]\n\n")

    # faccio data cleaning sul dataset
    cleaned_dataset = data_cleaning(dataset)
    # final_dataset_creation(balanced_dataset)


def data_cleaning(dataset):
    file.write("[FASE 1] DATA CLEANING\n\n")

    # elimino i duplicati
    dataset = dataset.drop_duplicates()

    # elimino le feature "company" e "agent" dato che producono molti valori null,
    dataset = dataset.drop(columns=["company", "agent"], axis=1)

    # elimino la feature "reservation_status_date" dato che serve poco al nostro scopo
    dataset = dataset.drop(columns=["reservation_status_date"], axis=1)

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

    balanced_count = dataset["is_canceled"].value_counts()
    print("\nIl bilanciamento dopo aver tolto i duplicati è: \n" + str(balanced_count) + "\n")
    utils.create_pie_chart(dataset, "Il bilanciamento del dataset dopo il data cleaning è:",
                           "bilanciamento_rimozione_duplicati")

    balanced_count = dataset["is_canceled"].value_counts()
    print("\nIl bilanciamento del dataset dopo il data cleaning è: \n" + str(balanced_count) + "\n")

    return dataset


def final_dataset_creation(dataset):
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
