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
    model = utils.load_obj(
        "src/classifier/modello_finale.sav")
    scaler = utils.load_obj("src/classifier/scaler.sav")
    encoder = utils.load_obj(
        "src/classifier/encoder.sav")
    data = pd.DataFrame(columns=['hotel', 'lead_time', 'adults',
                                 'children', 'meal', 'country',
                                 'market_segment', 'distribution_channel',
                                 'previous_cancellations', 'previous_bookings_not_canceled', 'reserved_room_type',
                                 'assigned_room_type', 'booking_changes', 'customer_type', 'adr',
                                 'required_car_parking_spaces', 'total_of_special_requests', 'stays_nights',
                                 'days_in_waiting_list', 'arrival_date_week_number',
                                 'arrival_date_day_of_month'])

    request_data = request.get_json()

    hotel = request_data['hotel']
    lead_time = request_data['lead_time']
    arrival_date_week_number = request_data['arrival_date_week_number']
    arrival_date_day_of_month = request_data['arrival_date_day_of_month']
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
    stays_nights = request_data['stays_in_week_nights'] + request_data['stays_in_weekend_nights']

    row = [hotel, lead_time, adults, children, meal,country, market_segment,
           distribution_channel, previous_cancellations, previous_bookings_not_canceled,
           reserved_room_type, assigned_room_type, booking_changes, customer_type, adr,
           required_car_parking_spaces, total_of_special_requests, stays_nights, days_in_waiting_list,
           arrival_date_week_number, arrival_date_day_of_month]

    data.loc[len(data)] = row

    # applico gli step necessari per dare in pasto la riga all'algoritmo di ML
    # scalo le feature da scalare
    filter = ['lead_time', 'days_in_waiting_list', 'adr', 'arrival_date_week_number', 'arrival_date_day_of_month']
    data[filter] = scaler.transform(data[filter])

    # converto in numeriche le variabili categoriche
    data = utils.convert_categorical(data, encoder, True)

    data.drop("days_in_waiting_list", axis=1, inplace=True)
    data.drop("arrival_date_week_number", axis=1, inplace=True)
    data.drop("arrival_date_day_of_month", axis=1, inplace=True)

    pred = model.predict(data)

    if pred == 0:
        result = '{"response":false}'
    else:
        result = '{"response":true}'

    return result


if __name__ == '__main__':
    app.run(debug=True)
