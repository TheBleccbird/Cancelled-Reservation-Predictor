from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from src.model_creation.steps import utils


def data_modeling_evaluation(dataset):
    # dataset = pd.read_csv("E:/final_dataset.csv")
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
    print(search.best_params_)

    # istanziamo un nuovo classificatore con i parametri suggeriti dal gs
    rf_classifier = RandomForestClassifier(criterion="log_loss", max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=9)
    rf_classifier.fit(X_train, y_train)

    rf_prediction = rf_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, rf_prediction)
    recall = recall_score(y_test, rf_prediction)
    matrix = confusion_matrix(y_test, rf_prediction)

    print("L'accuratezza del Random Forest sui dati di test è : " + str(accuracy))
    print("La recall del Random Forest sui dati di test è : " + str(recall))
    print("La matrice di confusione del Random Forest sui dati di test è : " + str(matrix))
    utils.create_confusion_matrix(matrix)

    # salvo il modello
    utils.save_obj(rf_classifier)
