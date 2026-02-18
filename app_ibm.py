# ==================== FLASK & WEB ====================
from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv

# ==================== DATA HANDLING ====================
import numpy as np
import pandas as pd

# ==================== DATA VISUALIZATION ====================
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== STATISTICAL ANALYSIS ====================
from scipy import stats

# ==================== DATA PREPROCESSING ====================
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ==================== DATASET SPLITTING ====================
from sklearn.model_selection import train_test_split

# ==================== MACHINE LEARNING MODELS ====================
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
try:
    import xgboost as xgb
except ImportError:
    xgb = None

# ==================== MODEL EVALUATION ====================
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# ==================== MODEL SAVING ====================
import pickle

# ==================== WARNINGS HANDLING ====================
import warnings
warnings.filterwarnings('ignore')

# Load environment variables for IBM Cloud
load_dotenv()

app = Flask(__name__)

# Load the trained model
try:
    with open('payments.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    print("✓ Model loaded successfully!")
except FileNotFoundError:
    print("✗ payments.pkl not found. Please ensure the model file is in the correct directory.")
    model = None


# IBM Cloud / Watson Studio Integration (Optional)
# Uncomment and configure with your IBM credentials
"""
from ibm_watson_machine_learning.client import APIClient

# IBM ML credentials
IBM_CREDENTIALS = {
    'url': os.getenv('IBM_ML_URL'),
    'username': os.getenv('IBM_ML_USERNAME'),
    'password': os.getenv('IBM_ML_PASSWORD'),
    'instance_id': os.getenv('IBM_ML_INSTANCE_ID')
}

client = APIClient(IBM_CREDENTIALS)
"""


@app.route('/')
def home():
    """Landing page"""
    return render_template('home.html')


@app.route('/predict')
def predict_page():
    """User enters transaction details"""
    return render_template('predict.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction using local or IBM model"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get data from request
        data = request.get_json()
        
        # Prepare features for prediction
        features = np.array([data.get('features', [])])
        
        # Make prediction with local model
        prediction = model.predict(features)
        probability = model.predict_proba(features)[0]
        
        is_fraud = int(prediction[0])
        fraud_probability = float(probability[1]) if len(probability) > 1 else 0
        
        return jsonify({
            'prediction': is_fraud,
            'probability': fraud_probability,
            'message': 'Fraudulent Transaction' if is_fraud else 'Legitimate Transaction'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/submit', methods=['POST'])
def submit():
    """Display fraud / non-fraud result"""
    try:
        data = request.get_json()
        result = data.get('result')
        return render_template('submit.html', result=result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    # For IBM Cloud deployment, use PORT environment variable
    port = int(os.getenv('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
