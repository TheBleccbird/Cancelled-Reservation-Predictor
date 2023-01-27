from matplotlib import pyplot as plt
import seaborn as sns


def create_pie_chart(dataset, title, file_title):
    count = dataset["is_canceled"].value_counts()

    data = [count[1], count[0]]
    labels = ['Cancellate', 'Rispettate']
    colors = ["red", "green"]

    plt.pie(data, labels=labels, colors=colors, autopct='%.0f%%')

    plt.title(title)
    plt.show()
    plt.savefig("src/images/" + file_title + ".png")


def detect_outliers(dataset, column):
    quartile_1 = dataset[column].quantile(0.25)
    quartile_3 = dataset[column].quantile(0.75)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)

    return lower_bound, upper_bound


def create_box_plot(dataset, column):
    sns.boxplot(dataset[column])
    plt.title(column)
    plt.show()


def create_graph(dataset, column):
    cancelled = dataset[dataset["is_canceled"] == 1]
    not_canceled = dataset[dataset["is_canceled"] == 0]

    plt.hist([cancelled[column], not_canceled[column]], label=['canceled', 'not canceled'],
             color=['red', 'green'])
    plt.legend(loc='upper right')

    plt.ylabel('count')
    plt.xlabel(column)
    plt.title("Feature: " + column)
    plt.show()


def create_count_plot(dataset, x, y):
    sns.countplot(dataset, x=x, hue=y)
    plt.show()
