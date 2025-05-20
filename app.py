import pickle
import numpy as np  # type: ignore
from flask import Flask, render_template, request, redirect, url_for  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from keras.models import load_model  # type: ignore

# === Paths (Updated to match train.py output) ===
MODEL_PATH = 'models/lung_cancer_model_final.keras'
SCALER_PATH = 'models/scaler_final.pkl'
FEATURES_PATH = 'models/training_features_final.pkl' 

app = Flask(__name__)

# Load the trained machine learning model
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading Keras model: {e}")
    # Handle error appropriately, e.g., exit or use a fallback
    model = None # Or raise an error

# Load the scaler object used during training
try:
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    print(f"Error: Scaler file not found at {SCALER_PATH}")
    # Handle error: maybe exit, or if scaling is optional for some flows, set scaler to None
    scaler = None # Or raise an error
except Exception as e:
    print(f"Error loading scaler: {e}")
    scaler = None # Or raise an error


# Define the order of features as used in training (based on your list)
# This order is CRUCIAL and must match the order of columns the model and scaler expect.
TRAINING_FEATURES_ORDER = [
    'Obesity', 'GeneticRisk', 'ChronicLungDisease', 'BalancedDiet', 'SwallowingDiff',
    'Fatigue', 'Wheezing', 'WeightLoss', 'DustAllergy', 'CoughingBlood',
    'Gender', 'AirPollution', 'ChestPain', 'Smoking', 'DryCough'
]

# Mapping from the training feature names to HTML form field names
# This helps if your form names are slightly different or for clarity.
# For this case, they mostly match, but 'SwallowingDiff' -> 'swallowing_difficulty'
FORM_FIELD_MAP = {
    'Obesity': 'obesity',
    'GeneticRisk': 'genetic_risk',
    'ChronicLungDisease': 'chronic_lung_disease',
    'BalancedDiet': 'balanced_diet',
    'SwallowingDiff': 'swallowing_difficulty', # HTML form uses 'swallowing_difficulty'
    'Fatigue': 'fatigue',
    'Wheezing': 'wheezing',
    'WeightLoss': 'weight_loss',
    'DustAllergy': 'dust_allergy',
    'CoughingBlood': 'coughing_blood',
    'Gender': 'gender',
    'AirPollution': 'air_pollution',
    'ChestPain': 'chest_pain',
    'Smoking': 'smoking',
    'DryCough': 'dry_cough'
}


# Define the home page route


@app.route('/')
def home():
    # Assuming your favicon is in static/images/
    # If not, adjust the path or remove if not needed here.
    # favicon_url = url_for('static', filename='images/favicon.png')
    # return render_template('home.html', favicon_url=favicon_url)
    return render_template('home.html') # Simpler if favicon is standardly linked in HTML

# Define the form page route
@app.route('/form')
def form_page(): # Renamed function to avoid conflict with 'form' from flask.request
    return render_template('form.html')

@app.route('/results', methods=['GET', 'POST'])
def results():
    if request.method == 'POST':
        if not model or not scaler:
            return "Error: Model or Scaler not loaded. Please check server logs.", 500

        # Print out the form data for debugging
        print("Form Data Received:", request.form)

        input_data = []
        try:
            for feature_name in TRAINING_FEATURES_ORDER:
                form_field_name = FORM_FIELD_MAP[feature_name]
                value_str = request.form.get(form_field_name)

                if value_str is None:
                    # Handle missing fields if necessary, or raise an error
                    print(f"Error: Missing form field {form_field_name} for feature {feature_name}")
                    return f"Error: Missing data for {form_field_name}", 400

                if feature_name == 'Gender':
                    # HTML form sends 'male' or 'female'
                    if value_str.lower() == 'male':
                        input_data.append(1)
                    elif value_str.lower() == 'female':
                        input_data.append(2)
                    else:
                        # Handle unexpected gender value
                        print(f"Error: Invalid value for Gender: {value_str}")
                        return f"Error: Invalid value for Gender: {value_str}", 400
                else:
                    # All other features are expected to be numbers (1-9 range as per form)
                    input_data.append(int(value_str))
        except ValueError as e:
            print(f"Error converting form data to integer: {e}")
            return "Error: Invalid input data type. Please ensure all fields are numbers.", 400
        except KeyError as e:
            print(f"Error: Feature mapping issue: {e}")
            return "Error: Internal server configuration error regarding feature mapping.", 500


        # Prepare the data for prediction
        # The input_data list is now correctly ordered and has 15 features
        new_data_np = np.array([input_data], dtype=float) # Ensure dtype is float for scaler

        print("Data prepared for scaling:", new_data_np)

        # Standardize the new data using the loaded scaler
        # The scaler must have been fitted on 15 features in the same order
        try:
            new_data_scaled = scaler.transform(new_data_np)
        except ValueError as e:
            # This can happen if the number of features in new_data_np (15)
            # does not match the number of features the scaler was trained on.
            print(f"Error during scaling: {e}. Check if scaler was trained on {len(TRAINING_FEATURES_ORDER)} features.")
            return f"Error during data scaling. Ensure model and scaler are for {len(TRAINING_FEATURES_ORDER)} features.", 500

        print("Data after scaling:", new_data_scaled)

        # Make predictions
        predictions = model.predict(new_data_scaled)
        print("Raw predictions from model:", predictions)

        # Convert the predictions to class labels (0 for Low, 1 for Medium, 2 for High)
        predicted_classes = np.argmax(predictions, axis=1)
        predicted_class = predicted_classes[0]

        # Determine the predicted outcome based on the prediction
        if predicted_class == 0:
            outcome = 'Low'
        elif predicted_class == 1:
            outcome = 'Medium'
        else: # Assuming class 2 is High and no other classes exist
            outcome = 'High'

        print(f"Predicted Class: {predicted_class}, Outcome: {outcome}")

        # Render the results template with the predicted outcome
        return render_template('results.html', outcome=outcome)
    else:
        # If the request method is not POST, redirect to the home page
        return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
