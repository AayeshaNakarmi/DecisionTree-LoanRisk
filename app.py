from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the trained model
model_filename = 'decision_tree_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Loan Risk Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json(force=True)

    # Extract features from the JSON
    initial_payment = data.get('Initial payment')
    last_payment = data.get('Last payment')
    credit_score = data.get('Credit Score')
    house_number = data.get('House Number')

    # Check if all features are provided
    if initial_payment is None or last_payment is None or credit_score is None or house_number is None:
        return jsonify({'error': 'Missing feature(s) in the request'}), 400

    # Convert data into numpy array
    data_array = np.array([[initial_payment, last_payment, credit_score, house_number]])
    
    # Make prediction using the loaded model
    prediction = model.predict(data_array)
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
