import joblib
import seaborn as sns
from matplotlib import pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn.preprocessing import LabelEncoder


def create_pie_chart(dataset, title, file_title):
    plt.rcParams.update(plt.rcParamsDefault)
    count = dataset["is_canceled"].value_counts()

    data = [count[1], count[0]]
    labels = ['Cancellate', 'Rispettate']
    colors = ["red", "green"]

    plt.pie(data, labels=labels, colors=colors, autopct='%.0f%%')

    plt.title(title)
    plt.savefig("src/images/" + file_title + ".png")
    plt.show()


def detect_outliers(dataset, column):
    quartile_1 = dataset[column].quantile(0.25)
    quartile_3 = dataset[column].quantile(0.75)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)

    return lower_bound, upper_bound


def create_evaluation_plot(accuracys, title):
    plt.rcParams['figure.figsize'] = [14, 8]

    plt.figure()
    plt.xlabel("Numero di feature")
    plt.ylabel("Accuracy (%)")

    fig, ax = plt.subplots()

    y1 = accuracys[0]
    y2 = accuracys[1]
    y3 = accuracys[2]
    y4 = accuracys[3]
    y5 = accuracys[4]

    ax.plot(y1, 'rs-', label='Random Forest')
    ax.plot(y2, 'bs-', label='Naive Bayes')
    ax.plot(y3, 'gs-', label='Decision Tree')
    ax.plot(y4, 'ys-', label='K Neighbors')
    ax.plot(y5, 'cs-', label='Logistic Regression')

    # leg = ax.legend(loc=2, bbox_to_anchor=(1.0, 1.0))
    plt.savefig('src/images/accuracys', bbox_inches='tight')

    plt.title(title)
    plt.show()


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
    plt.savefig("src/images/" + column + ".png")
    plt.show()


def create_grid_numeric(dataset, ):
    plt.rcParams['figure.figsize'] = [14, 7]
    fig, axs = plt.subplots(4, 4)

    cancelled = dataset[dataset["is_canceled"] == 1]
    not_canceled = dataset[dataset["is_canceled"] == 0]

    axs[0, 0].hist([cancelled["lead_time"], not_canceled["lead_time"]], label=['canceled', 'not canceled'],
                   color=['red', 'green'])
    axs[0, 0].set_title("lead_time", fontsize=10)

    axs[0, 1].hist([cancelled["stays_in_weekend_nights"], not_canceled["stays_in_weekend_nights"]],
                   label=['canceled', 'not canceled'], color=['red', 'green'])
    axs[0, 1].set_title("stays_in_weekend_nights", fontsize=10)

    axs[0, 2].hist([cancelled["stays_in_week_nights"], not_canceled["stays_in_week_nights"]],
                   label=['canceled', 'not canceled'], color=['red', 'green'])
    axs[0, 2].set_title("stays_in_week_nights", fontsize=10)

    axs[0, 3].hist([cancelled["adults"], not_canceled["adults"]], label=['canceled', 'not canceled'],
                   color=['red', 'green'])
    axs[0, 3].set_title("adults", fontsize=10)

    axs[1, 0].hist([cancelled["babies"], not_canceled["babies"]], label=['canceled', 'not canceled'],
                   color=['red', 'green'])
    axs[1, 0].set_title("babies", fontsize=10)

    axs[1, 1].hist([cancelled["children"], not_canceled["children"]], label=['canceled', 'not canceled'],
                   color=['red', 'green'])
    axs[1, 1].set_title("children", fontsize=10)

    axs[1, 2].hist([cancelled["previous_cancellations"], not_canceled["previous_cancellations"]],
                   label=['canceled', 'not canceled'], color=['red', 'green'])
    axs[1, 2].set_title("previous_cancellations", fontsize=10)

    axs[1, 3].hist([cancelled["previous_bookings_not_canceled"], not_canceled["previous_bookings_not_canceled"]],
                   label=['canceled', 'not canceled'], color=['red', 'green'])
    axs[1, 3].set_title("previous_bookings_not_canceled", fontsize=10)

    axs[2, 0].hist([cancelled["booking_changes"], not_canceled["booking_changes"]], label=['canceled', 'not canceled'],
                   color=['red', 'green'])
    axs[2, 0].set_title("booking_changes", fontsize=10)

    axs[2, 1].hist([cancelled["days_in_waiting_list"], not_canceled["days_in_waiting_list"]],
                   label=['canceled', 'not canceled'], color=['red', 'green'])
    axs[2, 1].set_title("days_in_waiting_list", fontsize=10)

    axs[2, 2].hist([cancelled["adr"], not_canceled["adr"]], label=['canceled', 'not canceled'], color=['red', 'green'])
    axs[2, 2].set_title("adr", fontsize=10)

    axs[2, 3].hist([cancelled["total_of_special_requests"], not_canceled["total_of_special_requests"]],
                   label=['canceled', 'not canceled'], color=['red', 'green'])
    axs[2, 3].set_title("total_of_special_requests", fontsize=10)

    axs[3, 0].hist([cancelled["required_car_parking_spaces"], not_canceled["required_car_parking_spaces"]],
                   label=['canceled', 'not canceled'], color=['red', 'green'])
    axs[3, 0].set_title("required_car_parking_spaces", fontsize=10)

    axs[3, 1].hist([cancelled["arrival_date_week_number"], not_canceled["arrival_date_week_number"]],
                   label=['canceled', 'not canceled'], color=['red', 'green'])
    axs[3, 1].set_title("arrival_date_week_number", fontsize=10)

    axs[3, 2].hist([cancelled["arrival_date_day_of_month"], not_canceled["arrival_date_day_of_month"]],
                   label=['canceled', 'not canceled'], color=['red', 'green'])
    axs[3, 2].set_title("arrival_date_day_of_month", fontsize=10)

    """axs[1, 0].sharex(axs[0, 0])
    axs[0, 1].plot(x + 1, y + 1)
    axs[0, 1].set_title("unrelated")
    axs[1, 1].plot(x + 2, y + 2)
    axs[1, 1].set_title("also unrelated")"""

    fig.tight_layout()
    plt.savefig("src/images/" + "numerici" + ".png")
    plt.show()


def create_grid_enums(dataset, ):
    plt.rcParams['figure.figsize'] = [16, 16]

    fig, axs = plt.subplots(3, 4)

    sns.countplot(dataset, x="is_canceled", hue="hotel", ax=axs[0, 0])
    sns.countplot(dataset, x="is_canceled", hue="arrival_date_year", ax=axs[0, 1])
    sns.countplot(dataset, x="is_canceled", hue="arrival_date_month", ax=axs[0, 2])
    sns.countplot(dataset, x="is_canceled", hue="meal", ax=axs[0, 3])
    sns.countplot(dataset, x="is_canceled", hue="market_segment", ax=axs[1, 0])
    sns.countplot(dataset, x="is_canceled", hue="distribution_channel", ax=axs[1, 1])
    sns.countplot(dataset, x="is_canceled", hue="reserved_room_type", ax=axs[1, 2])
    sns.countplot(dataset, x="is_canceled", hue="assigned_room_type", ax=axs[1, 3])
    sns.countplot(dataset, x="is_canceled", hue="deposit_type", ax=axs[2, 0])
    sns.countplot(dataset, x="is_canceled", hue="customer_type", ax=axs[2, 1])
    sns.countplot(dataset, x="is_canceled", hue="reservation_status", ax=axs[2, 2])

    fig.tight_layout()
    plt.savefig("src/images/" + "categorici" + ".png")
    plt.show()


def create_count_plot(dataset, x, y):
    sns.countplot(dataset, x=x, hue=y)

    if y == "arrival_date_month":
        plt.legend(loc="upper right")

    plt.savefig("src/images/" + y + ".png")
    plt.show()


def create_heat_map(dataset):
    plt.figure(figsize=(24, 12))
    sns.heatmap(dataset.corr(numeric_only=True), annot=True, linewidths=1)
    plt.savefig("src/images/correlazione.png")
    plt.show()


def create_confusion_matrix(confusion_matrix):
    plt.rcParams.update({'font.size': 16})
    fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix)
    plt.savefig("src/images/matrice_di_confusione.png")
    plt.show()


def save_obj(model, filename):
    joblib.dump(model, filename)


def load_obj(filename):
    model = joblib.load(filename)

    return model


def convert_categorical(dataset, le, take_encoder):
    if not take_encoder:
        le = LabelEncoder()

    mapping_rooms = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6,
                     'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12,
                     'N': 13,
                     'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19,
                     'U': 20,
                     'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25}

    dataset['hotel'] = dataset['hotel'].map({'Resort Hotel': 0, 'City Hotel': 1})
    dataset['meal'] = dataset['meal'].map({'BB': 0, 'FB': 1, 'HB': 2, 'SC': 3})
    dataset["customer_type"] = dataset['customer_type'].map(
        {'Contract': 0, 'Transient': 1, 'Transient-Party': 2, 'Group': 3})

    if take_encoder:
        dataset["country"] = le.transform(dataset["country"])
    else:
        dataset["country"] = le.fit_transform(dataset["country"])
        dataset['arrival_date_month'] = dataset['arrival_date_month'].map(
            {'January': 0, 'February': 1, 'March': 2, 'April': 3,
             'May': 4, 'June': 5, 'July': 6, 'August': 7
                , 'September': 8, 'October': 9, 'November': 10,
             'December': 11})

    dataset['market_segment'] = dataset['market_segment'].map(
        {'Corporate': 0, 'Direct': 1, "Groups": 2, 'Online TA': 3, 'Offline TA/TO': 4, 'Complementary': 5,
         'Aviation': 6})
    dataset['distribution_channel'] = dataset['distribution_channel'].map(
        {'Corporate': 0, 'Direct': 1, 'TA/TO': 2, 'GDS': 3})
    dataset["reserved_room_type"] = dataset['reserved_room_type'].map(mapping_rooms)
    dataset["assigned_room_type"] = dataset['assigned_room_type'].map(mapping_rooms)

    if take_encoder:
        return dataset
    else:
        return dataset, le
