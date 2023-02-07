from matplotlib import pyplot as plt
import seaborn as sns


def create_pie_chart(dataset, title, file_title):
    plt.rcParams.update(plt.rcParamsDefault)
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


def create_evaluation_plot(accurracy_list, ticks, title):
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.figure()
    plt.xticks(ticks)
    plt.yticks(accurracy_list)
    plt.xlabel("Numero di feature")
    plt.ylabel("Accuracy (%)")
    plt.plot(range(1, len(accurracy_list) + 1), accurracy_list)
    # plt.savefig('feature_auc_nselected.png', bbox_inches='tight', pad_inches=1)
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
    plt.show()


def create_grid_numeric(dataset, ):
    plt.rcParams['figure.figsize'] = [14, 7]
    fig, axs = plt.subplots(4, 4)

    cancelled = dataset[dataset["is_canceled"] == 1]
    not_canceled = dataset[dataset["is_canceled"] == 0]

    axs[0, 0].hist([cancelled["lead_time"], not_canceled["lead_time"]], label=['canceled', 'not canceled'], color=['red', 'green'])
    axs[0, 0].set_title("lead_time", fontsize=10)

    axs[0, 1].hist([cancelled["stays_in_weekend_nights"], not_canceled["stays_in_weekend_nights"]], label=['canceled', 'not canceled'], color=['red', 'green'])
    axs[0, 1].set_title("stays_in_weekend_nights", fontsize=10)

    axs[0, 2].hist([cancelled["stays_in_week_nights"], not_canceled["stays_in_week_nights"]], label=['canceled', 'not canceled'], color=['red', 'green'])
    axs[0, 2].set_title("stays_in_week_nights", fontsize=10)

    axs[0, 3].hist([cancelled["adults"], not_canceled["adults"]], label=['canceled', 'not canceled'], color=['red', 'green'])
    axs[0, 3].set_title("adults", fontsize=10)

    axs[1, 0].hist([cancelled["babies"], not_canceled["babies"]], label=['canceled', 'not canceled'], color=['red', 'green'])
    axs[1, 0].set_title("babies", fontsize=10)

    axs[1, 1].hist([cancelled["children"], not_canceled["children"]], label=['canceled', 'not canceled'], color=['red', 'green'])
    axs[1, 1].set_title("children", fontsize=10)

    axs[1, 2].hist([cancelled["previous_cancellations"], not_canceled["previous_cancellations"]], label=['canceled', 'not canceled'], color=['red', 'green'])
    axs[1, 2].set_title("previous_cancellations", fontsize=10)

    axs[1, 3].hist([cancelled["previous_bookings_not_canceled"], not_canceled["previous_bookings_not_canceled"]], label=['canceled', 'not canceled'], color=['red', 'green'])
    axs[1, 3].set_title("previous_bookings_not_canceled", fontsize=10)

    axs[2, 0].hist([cancelled["booking_changes"], not_canceled["booking_changes"]], label=['canceled', 'not canceled'], color=['red', 'green'])
    axs[2, 0].set_title("booking_changes", fontsize=10)

    axs[2, 1].hist([cancelled["days_in_waiting_list"], not_canceled["days_in_waiting_list"]], label=['canceled', 'not canceled'], color=['red', 'green'])
    axs[2, 1].set_title("days_in_waiting_list", fontsize=10)

    axs[2, 2].hist([cancelled["adr"], not_canceled["adr"]], label=['canceled', 'not canceled'], color=['red', 'green'])
    axs[2, 2].set_title("adr", fontsize=10)

    axs[2, 3].hist([cancelled["total_of_special_requests"], not_canceled["total_of_special_requests"]], label=['canceled', 'not canceled'], color=['red', 'green'])
    axs[2, 3].set_title("total_of_special_requests", fontsize=10)

    axs[3, 0].hist([cancelled["required_car_parking_spaces"], not_canceled["required_car_parking_spaces"]], label=['canceled', 'not canceled'], color=['red', 'green'])
    axs[3, 0].set_title("required_car_parking_spaces", fontsize=10)

    """axs[1, 0].sharex(axs[0, 0])
    axs[0, 1].plot(x + 1, y + 1)
    axs[0, 1].set_title("unrelated")
    axs[1, 1].plot(x + 2, y + 2)
    axs[1, 1].set_title("also unrelated")"""

    fig.tight_layout()
    plt.show()



def create_grid_enums(dataset, ):
    plt.rcParams['figure.figsize'] = [16, 16]

    fig, axs = plt.subplots(3, 4)

    sns.countplot(dataset,x="is_canceled", hue="hotel", ax=axs[0, 0])
    sns.countplot(dataset,x="is_canceled", hue="arrival_date_year", ax=axs[0, 1])
    sns.countplot(dataset,x="is_canceled", hue="arrival_date_month", ax=axs[0, 2])
    sns.countplot(dataset,x="is_canceled", hue="meal", ax=axs[0, 3])
    sns.countplot(dataset,x="is_canceled", hue="market_segment", ax=axs[1, 0])
    sns.countplot(dataset,x="is_canceled", hue="distribution_channel", ax=axs[1, 1])
    sns.countplot(dataset,x="is_canceled", hue="reserved_room_type", ax=axs[1, 2])
    sns.countplot(dataset,x="is_canceled", hue="assigned_room_type", ax=axs[1, 3])
    sns.countplot(dataset,x="is_canceled", hue="deposit_type", ax=axs[2, 0])
    sns.countplot(dataset,x="is_canceled", hue="customer_type", ax=axs[2, 1])
    sns.countplot(dataset,x="is_canceled", hue="reservation_status", ax=axs[2, 2])

    fig.tight_layout()
    plt.savefig("src/images/" + "categorici" + ".png")
    plt.show()

def create_count_plot(dataset, x, y):
    sns.countplot(dataset, x=x, hue=y)

    if y == "arrival_date_month":
        plt.legend(loc = "upper right")

    plt.savefig("src/images/" + y + ".png")
    plt.show()


def create_heat_map(dataset):
    plt.figure(figsize=(12, 8))
    sns.heatmap(dataset, annot=True)
    plt.show()
