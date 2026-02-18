from flask import Flask, render_template, request, jsonify
import os
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__, template_folder='templates')
app.secret_key = os.environ.get('SECRET_KEY', 'fraud-detection-dev-key-change-in-production')

# ------------------ LOAD MODEL ------------------

project_root = os.path.dirname(__file__)
MODEL_PATHS = []

env_path = os.environ.get('MODEL_PATH') or os.environ.get('PAYMENTS_PKL')
if env_path:
    MODEL_PATHS.append(env_path)

MODEL_PATHS.extend([
    os.path.join(project_root, 'payments.pkl'),
    os.path.join(os.getcwd(), 'payments.pkl'),
    os.path.join(project_root, 'Training', 'payments.pkl'),
])

model = None
for p in MODEL_PATHS:
    try:
        if p and os.path.exists(p):
            with open(p, 'rb') as f:
                model = pickle.load(f)
            print(f"Model loaded from: {p}")
            break
    except Exception as e:
        print(f"Failed to load model from {p}: {e}")

if model is None:
    print("Model not loaded. Place 'payments.pkl' correctly.")

# ------------------ PARSE INPUT ------------------

def parse_features_from_request(req):
    try:
        if req.is_json:
            payload = req.get_json()
            df = pd.DataFrame([payload])
        else:
            df = pd.DataFrame([{
                'step': float(req.form.get('step', 0)),
                'type': req.form.get('type', '').upper(),
                'amount': float(req.form.get('amount', 0)),
                'oldbalanceOrg': float(req.form.get('oldbalanceOrg', 0)),
                'newbalanceOrig': float(req.form.get('newbalanceOrig', 0)),
                'oldbalanceDest': float(req.form.get('oldbalanceDest', 0)),
                'newbalanceDest': float(req.form.get('newbalanceDest', 0)),
                'isFlaggedFraud': float(req.form.get('isFlaggedFraud', 0)),
            }])
    except Exception:
        return None

    # 🔥 FORCE CORRECT COLUMN ORDER
    expected_cols = [
        'step',
        'type',
        'amount',
        'oldbalanceOrg',
        'newbalanceOrig',
        'oldbalanceDest',
        'newbalanceDest',
        'isFlaggedFraud'
    ]

    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_cols]
    return df

# ------------------ ROUTES ------------------

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['GET'])
def predict_page():
    return render_template('predict.html')


@app.route('/predict', methods=['POST'])
def predict():

    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    X = parse_features_from_request(request)
    if X is None:
        return jsonify({'error': 'Invalid input'}), 400

    # If model is not pipeline → convert manually
    def _df_to_model_array(df):
        numeric_cols = [
            'step',
            'amount',
            'oldbalanceOrg',
            'newbalanceOrig',
            'oldbalanceDest',
            'newbalanceDest'
        ]
        type_cats = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']

        row = df.iloc[0]
        vals = []

        for c in numeric_cols:
            vals.append(float(row.get(c, 0)))

        t = str(row.get('type', '')).upper()
        for cat in type_cats:
            vals.append(1.0 if t == cat else 0.0)

        return np.array(vals).reshape(1, -1)

    if hasattr(model, 'named_steps'):
        X_model = X
    else:
        X_model = _df_to_model_array(X)

    # ------------------ HEURISTIC RULE ------------------

    try:
        row = X.iloc[0]
        ob_org = float(row['oldbalanceOrg'])
        amt = float(row['amount'])
        new_org = float(row['newbalanceOrig'])
        ob_dest = float(row['oldbalanceDest'])
        new_dest = float(row['newbalanceDest'])

        if (abs(ob_org - amt) < 1e-6 and
            new_org == 0.0 and
            ob_dest == 0.0 and
            abs(new_dest - amt) < 1e-6):

            return jsonify({
                'prediction': 1,
                'probability': 1.0,
                'reason': 'heuristic_exact_sweep'
            }), 200

    except Exception:
        pass

    # ------------------ ML PREDICTION ------------------

    try:
        proba = None

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_model)[0][1]

            # 🔥 Lower threshold to make fraud detection stronger
            pred = 1 if proba > 0.3 else 0
        else:
            pred = model.predict(X_model)[0]

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'}), 500

    prob = float(proba) if proba is not None else None

    result = {
        'prediction': int(pred),
        'probability': prob
    }

    if request.is_json:
        return jsonify(result)

    return render_template('submit.html', result=result)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    return predict()


@app.route('/submit', methods=['GET'])
def submit_page():
    return render_template('submit.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
