import pandas as pd


def data_preparation(dataset):
    cleaned_dataset = data_cleaning(dataset)
    # balanced_dataset = data_balancing(cleaned_dataset)
    # final_dataset_creation(balanced_dataset)


def data_cleaning(dataset):
    dataset = dataset.drop_duplicates()

    balanced_count = dataset["is_canceled"].value_counts()
    print("\nIl bilanciamento dopo aver tolto i duplicati è: \n" + str(balanced_count) + "\n")


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





