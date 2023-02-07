import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier


def data_modeling(dataset):
    # dataset = pd.read_csv("E:/final_dataset.csv")
    without_target = dataset.drop(columns=["is_canceled"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(without_target, dataset["is_canceled"],
                                                        test_size=0.20, random_state=42)

    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train, y_train)

    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X_train, y_train)

    dt_classifier = DecisionTreeClassifier()
    dt_classifier.fit(X_train, y_train)

    nb_prediction = nb_classifier.predict(X_test)
    rf_prediction = rf_classifier.predict(X_test)
    dt_prediction = dt_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, nb_prediction)
    recall = recall_score(y_test, nb_prediction)
    matrix = confusion_matrix(y_test, nb_prediction)

    print("L'accuratezza NB sui dati di test è : " + str(accuracy))
    print("La recall NB sui dati di test è : " + str(recall))
    print("La matrice di confusione NB sui dati di test è : " + str(matrix))

    accuracy = accuracy_score(y_test, rf_prediction)
    recall = recall_score(y_test, rf_prediction)
    matrix = confusion_matrix(y_test, rf_prediction)

    print("L'accuratezza RF sui dati di test è : " + str(accuracy))
    print("La recall RF sui dati di test è : " + str(recall))
    print("La matrice di confusione RF sui dati di test è : " + str(matrix))

    accuracy = accuracy_score(y_test, dt_prediction)
    recall = recall_score(y_test, dt_prediction)
    matrix = confusion_matrix(y_test, dt_prediction)

    print("L'accuratezza DT sui dati di test è : " + str(accuracy))
    print("La recall DT sui dati di test è : " + str(recall))
    print("La matrice di confusione DT sui dati di test è : " + str(matrix))
