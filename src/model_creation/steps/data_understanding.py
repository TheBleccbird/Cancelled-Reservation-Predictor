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
    utils.create_pie_chart(dataset, "Bilanciamento del dataset di partenza")

    # individuazione di possibili outlier
    file.write("\nDistribuzione dati per la feature [lead_time]\n")
    file.write(str(dataset.lead_time.describe()) + '\n')
    utils.detect_outliers(dataset, "lead_time")

    file.write("\nDistribuzione dati per la feature [stays_in_weekend_nights]\n")
    file.write(str(dataset.stays_in_weekend_nights.describe()) + '\n')
    utils.detect_outliers(dataset, "stays_in_weekend_nights")

    file.write("\nDistribuzione dati per la feature [stays_in_week_nights]\n")
    file.write(str(dataset.stays_in_week_nights.describe()) + '\n')
    utils.detect_outliers(dataset, "stays_in_week_nights")

    file.write("\nDistribuzione dati per la feature [adults]\n")
    file.write(str(dataset.adults.describe()) + '\n')
    utils.detect_outliers(dataset, "adults")

    file.write("\nDistribuzione dati per la feature [children]\n")
    file.write(str(dataset.children.describe()) + '\n')
    # utils.detect_outliers(dataset, "children")

    file.write("\nDistribuzione dati per la feature [babies]\n")
    file.write(str(dataset.babies.describe()) + '\n')
    utils.detect_outliers(dataset, "babies")

    file.write("\nDistribuzione dati per la feature [previous_cancellations]\n")
    file.write(str(dataset.previous_cancellations.describe()) + '\n')
    utils.detect_outliers(dataset, "previous_cancellations")

    file.write("\nDistribuzione dati per la feature [previous_bookings_not_canceled]\n")
    file.write(str(dataset.previous_bookings_not_canceled.describe()) + '\n')
    utils.detect_outliers(dataset, "previous_bookings_not_canceled")

    file.write("\nDistribuzione dati per la feature [booking_changes]\n")
    file.write(str(dataset.booking_changes.describe()) + '\n')
    utils.detect_outliers(dataset, "booking_changes")

    file.write("\nDistribuzione dati per la feature [days_in_waiting_list]\n")
    file.write(str(dataset.days_in_waiting_list.describe()) + '\n')
    utils.detect_outliers(dataset, "days_in_waiting_list")

    file.write("\nDistribuzione dati per la feature [adr]\n")
    file.write(str(dataset.adr.describe()) + '\n')
    utils.detect_outliers(dataset, "adr")

    file.write("\nDistribuzione dati per la feature [total_of_special_requests]\n")
    file.write(str(dataset.total_of_special_requests.describe()) + '\n')
    utils.detect_outliers(dataset, "total_of_special_requests")

    file.close()

    utils.create_graph(dataset, "lead_time")
    utils.create_graph(dataset, "stays_in_weekend_nights")
    utils.create_graph(dataset, "stays_in_week_nights")
    utils.create_graph(dataset, "adults")
    utils.create_graph(dataset, "babies")
    utils.create_graph(dataset, "previous_cancellations")
    utils.create_graph(dataset, "previous_bookings_not_canceled")
    utils.create_graph(dataset, "booking_changes")
    utils.create_graph(dataset, "days_in_waiting_list")
    utils.create_graph(dataset, "adr")
    utils.create_graph(dataset, "total_of_special_requests")

    return dataset
