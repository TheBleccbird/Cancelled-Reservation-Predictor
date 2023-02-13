import pandas as pd
import steps.utils as utils


def data_understanding():
    pd.set_option("display.max.columns", None)

    dataset = pd.read_csv("src/model_creation/dataset/hotel_bookings.csv", sep=";")

    file = open("src/logs/data_understanding.txt", "a")

    file.write("[FASE DI DATA UNDERSTANDING]\n\n")
    file.write(str(dataset.info()))

    n_duplicati = dataset.duplicated().sum()
    file.write("\nI duplicati nel dataset di partenza sono " + str(n_duplicati) + "\n")

    n_mancanti = dataset.isnull().sum()
    file.write("\nI dati mancanti nel dataset di partenza sono: \n" + str(n_mancanti) + "\n")

    balanced_count = dataset["is_canceled"].value_counts()
    file.write("\nIl bilanciamento del dataset di partenza è: \n" + str(balanced_count) + "\n")
    utils.create_pie_chart(dataset, "Bilanciamento del dataset di partenza", "bilanciamento_iniziale")

    # individuazione di possibili outlier
    file.write("\nDistribuzione dati per la feature [lead_time]\n")
    file.write(str(dataset.lead_time.describe()) + '\n')

    file.write("\nDistribuzione dati per la feature [stays_in_weekend_nights]\n")
    file.write(str(dataset.stays_in_weekend_nights.describe()) + '\n')

    file.write("\nDistribuzione dati per la feature [stays_in_week_nights]\n")
    file.write(str(dataset.stays_in_week_nights.describe()) + '\n')

    file.write("\nDistribuzione dati per la feature [adults]\n")
    file.write(str(dataset.adults.describe()) + '\n')

    file.write("\nDistribuzione dati per la feature [children]\n")
    file.write(str(dataset.children.describe()) + '\n')

    file.write("\nDistribuzione dati per la feature [babies]\n")
    file.write(str(dataset.babies.describe()) + '\n')

    file.write("\nDistribuzione dati per la feature [previous_cancellations]\n")
    file.write(str(dataset.previous_cancellations.describe()) + '\n')

    file.write("\nDistribuzione dati per la feature [previous_bookings_not_canceled]\n")
    file.write(str(dataset.previous_bookings_not_canceled.describe()) + '\n')

    file.write("\nDistribuzione dati per la feature [booking_changes]\n")
    file.write(str(dataset.booking_changes.describe()) + '\n')

    file.write("\nDistribuzione dati per la feature [days_in_waiting_list]\n")
    file.write(str(dataset.days_in_waiting_list.describe()) + '\n')

    file.write("\nDistribuzione dati per la feature [adr]\n")
    file.write(str(dataset.adr.describe()) + '\n')
    utils.detect_outliers(dataset, "adr")

    file.write("\nDistribuzione dati per la feature [total_of_special_requests]\n")
    file.write(str(dataset.total_of_special_requests.describe()) + '\n')

    file.write("\nAbbiamo notato che alcune prenotazioni non avevano clienti, ovvero 0 adulti, 0 bambini e 0 neonati. "
               "Il numero di record che hanno quest'anomalia è: " + str(len(dataset[(dataset["adults"] == 0) &
                                                                                    (dataset["children"] == 0) & (
                                                                                            dataset[
                                                                                                "babies"] == 0)])))

    file.write("\nAbbiamo notato che alcune prenotazioni avevano solo clienti non adulti, ovvero 0 adulti, almeno un "
               "bambino e un neonato. "
               "Il numero di record che hanno quest'anomalia è: " + str(len(dataset[(dataset["adults"] == 0) &
                                                                                    (dataset["children"] >= 0) & (
                                                                                            dataset[
                                                                                                "babies"] >= 0)])))

    # dati numerici
    utils.create_graph(dataset, "lead_time")
    utils.create_graph(dataset, "total_of_special_requests")
    utils.create_graph(dataset, "required_car_parking_spaces")

    # dati booleani
    utils.create_count_plot(dataset, "is_canceled", "is_repeated_guest")

    # Griglie per tutte le feature numeriche e categoriche
    utils.create_grid_numeric(dataset)
    utils.create_grid_enums(dataset)

    corr = dataset.corr(method='pearson', numeric_only=True)['is_canceled'][:]
    file.write("\nCorrelazione delle feature con is_canceled\n")
    file.write(str(corr) + '\n')
    print(str(corr) + '\n')

    utils.create_heat_map(dataset)

    file.close()
