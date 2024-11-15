from flask import Flask, request, jsonify
import joblib
import pandas as pd
import shap
import numpy as np 
from flask_cors import CORS
from datetime import datetime


app = Flask(__name__)
CORS(app)



# Load models and handle compatibility issues
model = joblib.load(r"C:\Users\laksh\prototype\backened\model.pkl")
label_encoders = joblib.load(r"C:\Users\laksh\prototype\backened\encoders.pkl")
fuel_model = joblib.load(r"C:\Users\laksh\prototype\backened\model4.pkl")
driver_model = joblib.load(r"C:\Users\laksh\prototype\backened\drive_score.pkl")




# Initialize SHAP explainer with the model
explainer = shap.Explainer(model)

@app.route('/')
def home():
    return "Welcome to the Vehicle Maintenance Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive and process input data
        data = request.get_json()
        df = pd.DataFrame([data])

        # Handle dates and encoding
        for date_column in ['Last_Service_Date', 'Warranty_Expiry_Date']:
            if date_column in df.columns:
                df[date_column] = (datetime.now() - pd.to_datetime(df[date_column], format="%d-%m-%Y")).dt.days
        for column, le in label_encoders.items():
            if column in df.columns and column not in ['Last_Service_Date', 'Warranty_Expiry_Date']:
                df[column] = le.transform(df[column])

        # Make prediction
        prediction = model.predict(df)[0]

        # Generate SHAP explanation
        shap_values = explainer(df)
        explanation_columns = ['Battery_Status', 'Brake_Condition', 'Tire_Condition', 'Maintenance_History', 'Reported_Issues']
        shap_explanation = []

        # Gather SHAP values with feature names and sort them
        for feature in explanation_columns:
            if feature in df.columns:
                feature_value = df[feature].iloc[0]
                if feature in label_encoders:
                    feature_value = label_encoders[feature].inverse_transform([feature_value])[0]
                shap_value = shap_values[:, :, 0].values[0, df.columns.get_loc(feature)]
                shap_explanation.append({"feature": feature.replace("_", " "), "value": feature_value, "shap_value": shap_value})
        
        # Sort SHAP explanation by absolute SHAP value in descending order
        shap_explanation = sorted(shap_explanation, key=lambda x: abs(x['shap_value']), reverse=True)

        # Return prediction and sorted SHAP values
        response = {
            'prediction': "Maintenance Needed" if prediction == 1 else "No Maintenance Needed",
            'explanation': shap_explanation  # List of dictionaries with feature name, value, and SHAP value
        }
        return jsonify(response)
    except Exception as e:
        print("An error occurred:", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/predict-fuel', methods=['POST'])
def predict_fuel():
    try:
        # Get JSON data from the request
        data = request.get_json()
        
        # Convert data into a DataFrame for model input
        df = pd.DataFrame([data])
        
        # Make prediction
        prediction = fuel_model.predict(df)[0]  # Replace with correct prediction logic
        co2_emission = round(prediction, 2)
        
        # Return prediction as JSON
        return jsonify({'prediction': co2_emission})
    
    except Exception as e:
        # Print error and return it as JSON
        print("Error occurred:", str(e))
        return jsonify({'error': str(e)}), 500

    except Exception as e:
        # Print the error for debugging and return a JSON error message
        print("An error occurred:", str(e))
        return jsonify({'error': str(e)}), 500



@app.route('/driver-score', methods=['POST'])
def driver_score():
    try:
        # Receive the JSON data from the request
        data = request.get_json()
        print("Received data:", data)  # Log received data for debugging

        # Convert JSON data to DataFrame
        df = pd.DataFrame([data])
        print("DataFrame created:\n", df)

        # Calculate the driver score
        score = driver_model.predict(df)[0]

        # Provide explanations based on score thresholds
        explanation = []
        if score < 60:
            explanation = [
                "Frequent sharp acceleration changes detected.",
                "Multiple sharp turns indicating abrupt driving style."
            ]
        elif score < 80:
            explanation = [
                "Moderate acceleration changes observed.",
                "Some sharp turns indicating careful but occasionally abrupt driving."
            ]
        else:
            explanation = [
                "Smooth acceleration observed.",
                "Minimal sharp turns, indicating steady driving style."
            ]

        if score is None:
            return jsonify({"error": "Calculation error"}), 500

        return jsonify({"score": score, "explanation": explanation})

    except Exception as e:
        print(f"Error in /driver-score endpoint: {e}")
        return jsonify({"error": str(e)}), 500




if __name__ == '__main__':
    app.run(debug=True)