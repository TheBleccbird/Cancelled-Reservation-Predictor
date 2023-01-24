import pandas as pd


def data_understanding():
    dataset = pd.read_csv("src/model_creation/dataset/reservations.csv")

    print(dataset.info())

    n_duplicati = dataset.duplicated().sum()
    print("\nI duplicati nel dataset di partenza sono " + str(n_duplicati))

    n_mancanti = dataset.isnull().sum()
    print("\nI dati mancanti nel dataset di partenza sono: \n" + str(n_mancanti))

    balanced_count = dataset["booking_status"].value_counts()
    print("\nIl bilanciamento del dataset di partenza è: \n" + str(balanced_count) + "\n")

    return dataset
