import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the saved model and preprocessing pipeline
model = joblib.load('best_xgboost_model.pkl')
pipeline = joblib.load('preprocessor_pipeline.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        print("Received data:", data)

        # Convert to DataFrame
        df = pd.DataFrame(data)
        print("DataFrame created:", df.head())

        # Preprocess the data
        df_preprocessed = pipeline.transform(df)
        print("Preprocessed data shape:", df_preprocessed.shape)

        # Ensure the model and data feature shapes match
        if df_preprocessed.shape[1] != model.n_features_in_:
            raise ValueError(f"Feature shape mismatch: Expected {model.n_features_in_}, got {df_preprocessed.shape[1]}")

        # Make predictions
        prediction = model.predict(df_preprocessed)
        prediction_proba = model.predict_proba(df_preprocessed)[:, 1]
        print("Prediction:", prediction)
        print("Prediction probability:", prediction_proba)

        # Create descriptive messages
        if prediction[0] == 1:
            message = "The patient is predicted to have heart disease."
        else:
            message = "The patient is predicted not to have heart disease."

        return jsonify({
            'prediction': prediction.tolist(),
            'prediction_proba': prediction_proba.tolist(),
            'message': message
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
