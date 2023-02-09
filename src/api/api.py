import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from flask import Flask, request
from src.model_creation.steps import utils
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return "Ops, non dovevi essere qui!"


@app.route('/predict', methods=['POST'])
def predict():
    model = utils.load_model()
    data = pd.DataFrame(columns=['hotel', 'lead_time', 'arrival_date_month', 'stays_in_weekend_nights',
                                 'stays_in_week_nights', 'adults', 'children', 'babies', 'meal', 'country',
                                 'market_segment', 'distribution_channel', 'is_repeated_guest',
                                 'previous_cancellations', 'previous_bookings_not_canceled', 'reserved_room_type',
                                 'assigned_room_type', 'booking_changes', 'deposit_type', 'days_in_waiting_list',
                                 'customer_type', 'adr', 'required_car_parking_spaces', 'total_of_special_requests'])

    request_data = request.get_json()

    hotel = request_data['hotel']
    lead_time = request_data['lead_time']
    arrival_date_month = request_data['arrival_date_month']
    stays_in_weekend_nights = request_data['stays_in_weekend_nights']
    stays_in_week_nights = request_data['stays_in_week_nights']
    adults = request_data['adults']
    children = request_data['children']
    babies = request_data['babies']
    meal = request_data['meal']
    country = request_data['country']
    market_segment = request_data['market_segment']
    distribution_channel = request_data['distribution_channel']
    is_repeated_guest = request_data['is_repeated_guest']
    previous_cancellations = request_data['previous_cancellations']
    previous_bookings_not_canceled = request_data['previous_bookings_not_canceled']
    reserved_room_type = request_data['reserved_room_type']
    assigned_room_type = request_data['assigned_room_type']
    booking_changes = request_data['booking_changes']
    deposit_type = request_data['deposit_type']
    days_in_waiting_list = request_data['days_in_waiting_list']
    customer_type = request_data['customer_type']
    adr = request_data['adr']
    required_car_parking_spaces = request_data['required_car_parking_spaces']
    total_of_special_requests = request_data['total_of_special_requests']

    row = [hotel, lead_time, arrival_date_month, stays_in_weekend_nights, stays_in_week_nights, adults, children,
           babies, meal,
           country, market_segment, distribution_channel, is_repeated_guest, previous_cancellations,
           previous_bookings_not_canceled, reserved_room_type, assigned_room_type, booking_changes, deposit_type,
           days_in_waiting_list, customer_type, adr, required_car_parking_spaces, total_of_special_requests]

    data.loc[len(data)] = row

    # applico gli step necessari per dare in pasto la riga all'algoritmo di ML
    # scalo le feature da scalare
    filter = ['lead_time', 'days_in_waiting_list', 'adr']
    scaler = MinMaxScaler()
    data[filter] = scaler.fit_transform(data[filter])

    # converto in numeriche le variabili categoriche
    le = LabelEncoder()

    data["hotel"] = le.fit_transform(data["hotel"])
    data["arrival_date_month"] = le.fit_transform(data["arrival_date_month"])
    data["meal"] = le.fit_transform(data["meal"])
    data["country"] = le.fit_transform(data["country"])
    data["market_segment"] = le.fit_transform(data["market_segment"])
    data["distribution_channel"] = le.fit_transform(data["distribution_channel"])
    data["reserved_room_type"] = le.fit_transform(data["reserved_room_type"])
    data["assigned_room_type"] = le.fit_transform(data["assigned_room_type"])
    data["deposit_type"] = le.fit_transform(data["deposit_type"])
    data["customer_type"] = le.fit_transform(data["customer_type"])

    pred = model.predict(data)

    if pred == 0:
        result = "False"
    else:
        result = "True"

    return result


if __name__ == '__main__':
    app.run(debug=True)
