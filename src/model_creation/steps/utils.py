from matplotlib import pyplot as plt
import seaborn as sns


def create_pie_chart(dataset, title):
    count = dataset["is_canceled"].value_counts()

    data = [count[1], count[0]]
    labels = ['Cancellate', 'Rispettate']
    colors = ["red", "green"]

    plt.pie(data, labels=labels, colors=colors, autopct='%.0f%%')

    plt.title(title)
    plt.show()


def detect_outliers(dataset, column):
    sns.boxplot(dataset[column])
    plt.title(column)
    plt.show()

    quartile_1 = dataset[column].quantile(0.25)
    quartile_3 = dataset[column].quantile(0.75)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    print("Lower bound:", lower_bound)
    print("Upper bound:", upper_bound)
    outliers = dataset[(dataset[column] <= lower_bound) | (dataset[column] >= upper_bound)]
    print("Numero di outlier: ", outliers.shape[0])

    count = outliers.is_canceled.value_counts()
    print("numero outlier prenotazioni confermate:", count[0])
    print("numero outlier prenotazioni cancellate:", count[1])


def create_graph(dataset, column):
    cancelled = dataset[dataset["is_canceled"] == 1]
    not_canceled = dataset[dataset["is_canceled"] == 0]

    plt.hist([cancelled[column], not_canceled[column]], label=['canceled', 'not canceled'],
             color=['red', 'green'])
    plt.legend(loc='upper right')

    plt.ylabel('numero istanze')
    plt.xlabel(column)
    plt.title("Feature: " + column)
    plt.show()
