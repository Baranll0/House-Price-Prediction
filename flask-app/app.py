from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Global variables to track the number of correct predictions and total predictions
correct_predictions = 0
total_predictions = 0

# Model, scaler ve R^2 skorunu yükleme
def load_model_and_scaler(city):
    model_path = f'../model/{city.lower()}.pickle'
    scaler_path = f'../scalers/{city.lower()}_scaler.pickle'
    r2_score_path = f'../model/{city.lower()}_r2_score.pickle'

    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)

    with open(r2_score_path, 'rb') as file:
        r2_score_value = pickle.load(file)

    return model, scaler, r2_score_value

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global correct_predictions, total_predictions

    city = request.form['city']
    beds = float(request.form['beds'])
    baths = float(request.form['baths'])
    square_feet = float(request.form['square_feet'])
    price_per_sqft = float(request.form['price_per_sqft'])
    hoa_month = float(request.form['hoa_month'])

    # Example: True value (In practice, this should be obtained from a source to compare predictions)
    true_value = float(request.form.get('true_value', 0))

    # Model ve scaler'ı yükle
    model, scaler, r2_score_value = load_model_and_scaler(city)

    # Kullanıcıdan alınan veriyi uygun formata getir
    input_data = np.array([[beds, baths, square_feet, price_per_sqft, hoa_month]])
    input_data_scaled = scaler.transform(input_data)

    # Tahmin yap
    prediction = model.predict(input_data_scaled)[0]

    # Update the count of total predictions
    total_predictions += 1

    # For demonstration purposes: Increment correct prediction count if prediction is close to true_value
    if abs(prediction - true_value) < 0.1:  # Adjust tolerance as needed
        correct_predictions += 1

    # Calculate accuracy
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0

    return render_template('result.html', prediction=prediction, accuracy=accuracy, correct_predictions=correct_predictions, total_predictions=total_predictions, r2_score=r2_score_value)

if __name__ == '__main__':
    app.run(debug=True)