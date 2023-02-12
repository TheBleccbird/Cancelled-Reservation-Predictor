import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from src.model_creation.steps import utils

file = open("src/logs/data_modeling_evaluation.txt", "a")


def data_modeling_evaluation():
    file.write("[FASE DI DATA MODELING ED EVALUATION]\n\n")
    file.write("[FASE 1] DATA MODELING\n\n")

    dataset = pd.read_csv(
        "src/model_creation/dataset/final_dataset.csv")
    without_target = dataset.drop(columns=["is_canceled"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(without_target, dataset["is_canceled"],
                                                        test_size=0.20, random_state=42)

    rf_classifier = RandomForestClassifier()

    parameters = {"criterion": ["gini", "entropy", "log_loss"],
                  'max_depth': [None, 3, 5, 7],
                  'min_samples_split': [2, 5, 10],
                  'min_samples_leaf': [1, 2, 4]
                  }

    search = HalvingGridSearchCV(rf_classifier, parameters, resource='n_estimators', max_resources=10, random_state=42).fit(X_train, y_train)

    # i migliori parametri tra quelli inseriti in "parameters"
    file.write("I migliori parametri secondo il metodo HalvingGridSearchCV sono: \n" + str(search.best_params_))

    # istanziamo un nuovo classificatore con i parametri suggeriti dal gs
    rf_classifier = RandomForestClassifier(criterion="log_loss", max_depth=None, min_samples_leaf=2,
                                           min_samples_split=10, n_estimators=9)
    rf_classifier.fit(X_train, y_train)

    rf_prediction = rf_classifier.predict(X_test)

    file.write("[FASE 2] EVALUATION\n\n")
    file.write("Report delle metriche di valutazione:\n")
    file.write(str(classification_report(y_test, rf_prediction)))

    matrix = confusion_matrix(y_test, rf_prediction)
    file.write("La matrice di confusione del Random Forest sui dati di test è : \n" + str(matrix))
    utils.create_confusion_matrix(matrix)

    # salvo il modello
    utils.save_obj(rf_classifier)
