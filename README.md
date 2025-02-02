﻿# House Price Prediction

A Flask-based web application for predicting house prices using machine learning models. Users can input house features to get a price prediction and view model performance metrics.

## Technologies Used

- **Flask**: Web framework for building the application.
- **Scikit-learn**: Machine learning library used for building and training the prediction model.
- **Bootstrap**: CSS framework for creating responsive and modern user interfaces.
- **Pickle**: Python module for serializing and deserializing Python objects (models and scalers).
- **HTML & CSS**: For creating the structure and styling of the web pages.

## Project Structure

- **app.py**: The main Flask application file.
- **model/**: Directory containing trained machine learning models.
    - `chicago.pickle`
- **scalers/**: Directory containing trained scaler objects.
    - `chicago_scaler.pickle`
- **templates/**: Directory containing HTML templates.
    - `index.html`
    - `result.html`
- **static/**: Directory containing static files like CSS.
    - `style.css`

## How to Run the Application

1. Clone the repository:
    ```bash
    git clone https://github.com/Baranll0/House-price-prediction.git
    cd house-price-prediction
    ```
2. Set up the environment:
    ```bash
    python -m venv venv
    ```

3. Activate the virtual environment:
    - On Windows:
      ```bash
      venv\Scripts\activate
      ```
    - On macOS/Linux:
      ```bash
      source venv/bin/activate
      ```

4. Install the required packages:
    ```bash
    pip install flask pandas scikit-learn
    ```
5. Go to file path:
    ```bash
    cd flask-app
    ```
6. Run the Flask application:
    ```bash
    python app.py
    ```

7. Open your browser and go to `http://127.0.0.1:5000` to use the application.

## Usage

1. On the home page, input the house features such as city, number of beds, baths, square feet, price per square foot, and HOA/month.
2. Click the "Predict" button to get the estimated house price.
3. The result page will display the predicted price and the R^2 score of the model.

## Model Training and Saving

The model is trained using the `RandomForestRegressor` from the scikit-learn library. The trained model and scaler are saved using the pickle module. The dataset used for training is obtained from Redfin.

## Future Improvements

- Enhance the model with more data for better accuracy.
- Add more cities and house features.
- Improve the user interface and user experience based on feedback.
