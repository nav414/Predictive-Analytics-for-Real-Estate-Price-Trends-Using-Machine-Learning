from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Custom function to format in Indian currency style
def format_in_indian_currency(value):
    value_str = str(int(value))
    # Split the string into the last three digits and the rest
    if len(value_str) > 3:
        last_three = value_str[-3:]
        remaining = value_str[:-3]
        # Insert commas in the remaining part every two digits from the right
        remaining = ','.join([remaining[max(i - 2, 0):i] for i in range(len(remaining), 0, -2)][::-1])
        formatted_value = remaining + ',' + last_three
    else:
        formatted_value = value_str
    return formatted_value

# Load the trained model and scaler
with open('chennai_prices_lgbm.pickle', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pickle', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', prediction='')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    int_sqft = float(request.form['INT_SQFT'])
    dist_mainroad = float(request.form['DIST_MAINROAD'])
    n_bedroom = int(request.form['N_BEDROOM'])
    n_bathroom = int(request.form['N_BATHROOM'])
    
    # Get the 'PARK_FACIL' value from the dropdown
    park_facil = 1 if request.form['PARK_FACIL'].strip().lower() == 'yes' else 0
    
    buildtype_house = 1 if request.form['BUILDTYPE'].strip().lower() == 'house' else 0
    buildtype_others = 1 if request.form['BUILDTYPE'].strip().lower() == 'others' else 0
    street_no_access = 1 if request.form['STREET'].strip().lower() == 'no access' else 0
    street_paved = 1 if request.form['STREET'].strip().lower() == 'paved' else 0
    area_encoded = {  # Adjust this based on your dummy variable encoding
        'AREA_Anna Nagar': 1 if request.form['AREA'].strip().lower() == 'anna nagar' else 0,
        'AREA_Chrompet': 1 if request.form['AREA'].strip().lower() == 'chrompet' else 0,
        'AREA_KK Nagar': 1 if request.form['AREA'].strip().lower() == 'kk nagar' else 0,
        'AREA_Karapakkam': 1 if request.form['AREA'].strip().lower() == 'karapakkam' else 0,
        'AREA_T Nagar': 1 if request.form['AREA'].strip().lower() == 't nagar' else 0,
        'AREA_Velachery': 1 if request.form['AREA'].strip().lower() == 'velachery' else 0,
    }

    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        'INT_SQFT': [int_sqft],
        'DIST_MAINROAD': [dist_mainroad],
        'N_BEDROOM': [n_bedroom],
        'N_BATHROOM': [n_bathroom],
        'PARK_FACIL_Yes': [park_facil],
        'BUILDTYPE_House': [buildtype_house],
        'BUILDTYPE_Others': [buildtype_others],
        'STREET_No Access': [street_no_access],
        'STREET_Paved': [street_paved],
        'AREA_Anna Nagar': [area_encoded['AREA_Anna Nagar']],
        'AREA_Chrompet': [area_encoded['AREA_Chrompet']],
        'AREA_KK Nagar': [area_encoded['AREA_KK Nagar']],
        'AREA_Karapakkam': [area_encoded['AREA_Karapakkam']],
        'AREA_T Nagar': [area_encoded['AREA_T Nagar']],
        'AREA_Velachery': [area_encoded['AREA_Velachery']],
    })

    # Scale the input data using the same scaler used for training data
    input_scaled = scaler.transform(input_data)

    # Make prediction using the trained model
    prediction = model.predict(input_scaled)[0]
    
    # Format the predicted price into Indian currency
    formatted_price = format_in_indian_currency(prediction)

    # Return the formatted prediction to the user
    return render_template('index.html', prediction=f'Predicted Price: ₹{formatted_price}')

if __name__ == '__main__':
    app.run(debug=True)


