from flask import Flask, request, jsonify, render_template, Response
import pandas as pd
import numpy as np
import io
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for rendering plots to files
import matplotlib.pyplot as plt
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define threshold levels for pollutants
thresholds = {
    'PM10': 50,
    'PM2.5': 25,
    'NO2': 40,
    'NH3': 35,
    'SO2': 20,
    'CO': 10,
    'O3': 100
}

# Define AQI breakpoints and ranges
breakpoints = {
    'PM2.5': [(0, 12), (12.1, 35.4), (35.5, 55.4), (55.5, 150.4), (150.5, 250.4), (250.5, 350.4), (350.5, 500.4)],
    'PM10': [(0, 54), (55, 154), (155, 254), (255, 354), (355, 424), (425, 504), (505, 604)],
    'NO2': [(0, 53), (54, 100), (101, 360), (361, 649), (650, 1249), (1250, 1649), (1650, 2049)],
    'CO': [(0, 4.4), (4.5, 9.4), (9.5, 12.4), (12.5, 15.4), (15.5, 30.4), (30.5, 40.4), (40.5, 50.4)],
    'SO2': [(0, 35), (36, 75), (76, 185), (186, 304), (305, 604), (605, 804), (805, 1004)],
    'O3': [(0, 54), (55, 70), (71, 85), (86, 105), (106, 200), (201, 300), (301, 400)],
    'NH3': [(0, 200), (201, 400), (401, 800), (801, 1200), (1201, 1800), (1801, 2400), (2401, 3000)]
}

aqi_ranges = [(0, 50), (51, 100), (101, 150), (151, 200), (201, 300), (301, 400), (401, 500)]

# Load data from the local CSV file once at startup
try:
    data = pd.read_csv('cleaned_data_filled.csv')
except FileNotFoundError:
    raise SystemExit("Error: 'cleaned_data_filled.csv' file not found.")
except pd.errors.EmptyDataError:
    raise SystemExit("Error: 'cleaned_data_filled.csv' file is empty.")
except pd.errors.ParserError:
    raise SystemExit("Error: 'cleaned_data_filled.csv' file is not in the correct format.")

# Function to calculate sub-index
def calculate_sub_index(concentration, breakpoints, aqi_ranges):
    for i, (bp_low, bp_high) in enumerate(breakpoints):
        if bp_low <= concentration <= bp_high:
            aqi_low, aqi_high = aqi_ranges[i]
            return ((concentration - bp_low) / (bp_high - bp_low)) * (aqi_high - aqi_low) + aqi_low
    return np.nan

# Helper function to create and save plots
def create_plot(x, y, xlabel, ylabel, title, highlight=None, highlight_color='red'):
    plt.figure(figsize=(15, 8))
    bars = plt.bar(x, y, color='skyblue')

    if highlight and highlight in x:
        selected_index = x.index(highlight)
        bars[selected_index].set_color(highlight_color)
        plt.text(highlight, y[selected_index], f'{y[selected_index]:.2f}',
                 ha='center', va='bottom', color='black', fontsize=12, fontweight='bold')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return img

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/new')
def new_page():
    return render_template('new.html')

@app.route('/states', methods=['GET'])
def get_states():
    try:
        states = data['state'].unique().tolist()
        return jsonify(states)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/cities', methods=['GET'])
def get_cities():
    try:
        state = request.args.get('state')
        if not state:
            return jsonify({"error": "State parameter is required"}), 400

        filtered_data = data[data['state'] == state]
        cities = filtered_data['city'].unique().tolist()

        if not cities:
            return jsonify({"error": "No cities found for the selected state"}), 404

        return jsonify(cities)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/check_pollution', methods=['GET'])
def check_pollution():
    try:
        city = request.args.get('city')
        state = request.args.get('state')

        place_data = data[(data['city'].str.lower() == city.lower()) & (data['state'] == state)]

        if place_data.empty:
            return jsonify({'error': 'City not found'}), 404

        result = {}
        for pollutant, threshold in thresholds.items():
            max_value = place_data.loc[place_data['pollutant_id'] == pollutant, 'pollutant_max'].max()
            max_value = 0 if pd.isna(max_value) else max_value
            result[pollutant] = 'Exceeded' if max_value > threshold else 'Safe'

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/plot_pollution', methods=['GET'])
def plot_pollution():
    state = request.args.get('state')
    city = request.args.get('city')

    state_data = data[data['state'] == state]
    city_pollution = state_data.groupby(['city', 'pollutant_id'])['pollutant_avg'].mean().unstack()

    for pollutant in breakpoints.keys():
        if pollutant in city_pollution.columns:
            city_pollution[pollutant + '_AQI'] = city_pollution[pollutant].apply(calculate_sub_index, args=(breakpoints[pollutant], aqi_ranges))

    city_pollution['Composite_AQI'] = city_pollution[[col for col in city_pollution.columns if '_AQI' in col]].max(axis=1)

    img = create_plot(
        x=city_pollution.index.tolist(),
        y=city_pollution['Composite_AQI'].tolist(),
        xlabel='City',
        ylabel='Composite AQI',
        title=f'Composite Pollution Index (AQI) by City in {state}',
        highlight=city,
        highlight_color='red'
    )
    return Response(img, mimetype='image/png')

@app.route('/plot_pollutant_levels', methods=['GET'])
def plot_pollutant_levels():
    state = request.args.get('state')
    city = request.args.get('city')
    pollutant = request.args.get('pollutant')

    state_data = data[data['state'] == state]
    city_pollutant_data = state_data[state_data['pollutant_id'] == pollutant].groupby('city')['pollutant_avg'].mean()

    img = create_plot(
        x=city_pollutant_data.index.tolist(),
        y=city_pollutant_data.tolist(),
        xlabel='City',
        ylabel=f'{pollutant} Level',
        title=f'{pollutant} Levels in Cities of {state}',
        highlight=city,
        highlight_color='red'
    )
    return Response(img, mimetype='image/png')

@app.route('/city_aqi', methods=['GET'])
def city_aqi():
    try:
        city = request.args.get('city')
        state = request.args.get('state')
        pollutant = request.args.get('pollutant')

        place_data = data[(data['city'].str.lower() == city.lower()) & (data['state'] == state)]

        if place_data.empty:
            return jsonify({'error': 'City not found'}), 404

        max_value = place_data.loc[place_data['pollutant_id'] == pollutant, 'pollutant_max'].max()
        max_value = 0 if pd.isna(max_value) else max_value
        sub_index = calculate_sub_index(max_value, breakpoints[pollutant], aqi_ranges)

        return jsonify({
            'city': city,
            'state': state,
            'pollutant': pollutant,
            'value': max_value,
            'sub_index': sub_index
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
