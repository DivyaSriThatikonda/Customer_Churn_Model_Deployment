from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the pickled Random Forest model
try:
    model = pickle.load(open('customer_churn.pkl', 'rb'))
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Home route to render the HTML page
@app.route('/')
def home():
    print("Home route accessed.")
    return render_template('index.html')

# Prediction route for handling form submission
@app.route('/predict', methods=['POST'])
def predict():
    print("Prediction route accessed.")
    
    # Collect data from form input fields
    try:
        tenure = float(request.form['tenure'])
        monthlycharges = float(request.form['monthlycharges'])
        print(f"Features received - Tenure: {tenure}, Monthly Charges: {monthlycharges}")
        
        # Create a feature array for prediction
        final_features = np.array([[tenure, monthlycharges]])
        
        # Make prediction using the model
        prediction = model.predict(final_features)
        print(f"Prediction made: {prediction}")

        # Translate prediction into a readable output
        output = 'Churn' if prediction[0] == 1 else 'No Churn'
        print(f"Prediction output: {output}")
        
        return render_template('index.html', prediction_text=f'The customer will: {output}')
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('index.html', prediction_text='Error in prediction.')

if __name__ == "__main__":
    print("Starting the Flask app...")
    app.run(debug=True)
