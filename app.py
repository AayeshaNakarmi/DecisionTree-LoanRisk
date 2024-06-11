from flask import Flask, render_template, request, jsonify
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
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from the form
    initial_payment = float(request.form['initial_payment'])
    last_payment = float(request.form['last_payment'])
    credit_score = int(request.form['credit_score'])
    house_number = int(request.form['house_number'])

    # Convert data into numpy array
    data_array = np.array([[initial_payment, last_payment, credit_score, house_number]])
    
    # Make prediction using the loaded model
    prediction = model.predict(data_array)
    
    # Render the prediction template with the result
    return render_template('prediction.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
