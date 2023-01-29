import pandas as pd
import steps.utils as utils


file = open("src/logs/data_understanding.txt", "a")

def data_preparation(dataset):
    file.write("[FASE DI DATA PREPARATION]\n\n")

    cleaned_dataset = data_cleaning(dataset)
    # final_dataset_creation(balanced_dataset)


def data_cleaning(dataset):

    file.write("[FASE 1] DATA CLEANING\n\n")

    # elimino i duplicati
    dataset = dataset.drop_duplicates()

    # elimino la feature "company" dato che produce molti valori null
    dataset = dataset.drop(columns=["company"], axis=1)

    # applico il most frequent imputation ai valori della feature "country" null
    dataset["country"] = dataset["country"].fillna(dataset['country'].value_counts().index[0])

    # elimino le righe con agent null    dataset = dataset.dropna()
    print(dataset.isnull().sum())

    # elimino le righe con 0 adulti, 0 bambini e 0 neonati
    dataset.drop(dataset[(dataset["adults"] == 0) & (dataset["children"] == 0) & (dataset["babies"] == 0)].index, inplace=True)

    # elimino le righe con 0 adulti, almeno un bambino o almeno un neonato
    dataset.drop(dataset[(dataset["adults"] == 0) & ((dataset["children"] >= 0) | (dataset["babies"] >= 0))].index, inplace=True)

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
    final_dataset = pd.DataFrame(columns=["label", "subject-email_from-message"])

    print("Inizio il salvataggio del nuovo csv")

    # inserisco tutti i nuovi dati nel file finale
    for i in range(len(dataset)):
        print("Generazione della riga: " + str(i) + "/" + str(len(dataset)))
        label = dataset.iloc[i]["label"]

        row = [label, subject + " " + email_from + " " + message]
        final_dataset.loc[len(finalDataset)] = row

    # converto in csv il DataFrame
    finalDataset.to_csv("src/model_creation/dataset/reservations_final.csv", index=False)





