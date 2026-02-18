import os
import pickle
import pandas as pd

MODEL_CANDIDATES = [
    os.environ.get('MODEL_PATH') or os.environ.get('PAYMENTS_PKL'),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'payments.pkl'),
    os.path.join(os.getcwd(), 'payments.pkl'),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Training', 'payments.pkl'),
]

model = None
for p in MODEL_CANDIDATES:
    if not p:
        continue
    if os.path.exists(p):
        try:
            with open(p, 'rb') as f:
                model = pickle.load(f)
            print('Loaded model from', p)
            break
        except Exception as e:
            print('Failed to load from', p, e)

if model is None:
    print('No model found. Checked:', MODEL_CANDIDATES)
    raise SystemExit(1)

# Example transaction (from your message)
row = {
    'step': 1,
    'amount': 200000.0,
    'oldbalanceOrg': 200000.0,
    'newbalanceOrig': 0.0,
    'oldbalanceDest': 0.0,
    'newbalanceDest': 200000.0,
    'isFlaggedFraud': 0,
    'type': 'TRANSFER'
}

df = pd.DataFrame([row])
print('\nInput DataFrame:')
print(df.to_dict(orient='records'))

try:
    pred = model.predict(df)
    print('\nModel.predict ->', pred)
except Exception as e:
    print('\nModel.predict error:', e)
    # If the model is a bare estimator (no pipeline), try building the numeric
    # input that was used during training (6 numeric + 5 one-hot `type`)
    try:
        numeric = [
            float(df.iloc[0].get('step', 0)),
            float(df.iloc[0].get('amount', 0)),
            float(df.iloc[0].get('oldbalanceOrg', 0)),
            float(df.iloc[0].get('newbalanceOrig', 0)),
            float(df.iloc[0].get('oldbalanceDest', 0)),
            float(df.iloc[0].get('newbalanceDest', 0)),
        ]
        type_cats = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
        t = str(df.iloc[0].get('type', '')).upper()
        onehot = [1.0 if t == c else 0.0 for c in type_cats]
        import numpy as np
        X = np.array(numeric + onehot).reshape(1, -1)
        print('\nModel.predict (after manual transform) ->', model.predict(X))
    except Exception as e2:
        print('\nManual-transform predict failed:', e2)

try:
    proba = model.predict_proba(df)
    print('\nModel.predict_proba ->', proba)
except Exception as e:
    print('\nModel.predict_proba error:', e)

# If pipeline expects numpy array shape, try transforming
try:
    from sklearn.pipeline import Pipeline
    if hasattr(model, 'predict') and isinstance(model, Pipeline):
        preprocessor = model.named_steps.get('preprocessor')
        if preprocessor is not None:
            X_trans = preprocessor.transform(df)
            print('\nTransformed shape:', getattr(X_trans, 'shape', 'n/a'))
except Exception as e:
    print('\nTransform check error:', e)
