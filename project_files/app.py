from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("payments.pkl", "rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict")
def predict():
    return render_template("predict.html")

@app.route("/submit", methods=["POST"])
@app.route("/submit", methods=["POST"])
def submit():

    step = float(request.form["step"])
    amount = np.log1p(float(request.form["amount"]))
    oldbalanceOrg = float(request.form["oldbalanceOrg"])
    newbalanceOrig = float(request.form["newbalanceOrig"])
    oldbalanceDest = float(request.form["oldbalanceDest"])
    newbalanceDest = float(request.form["newbalanceDest"])
    txn_type = request.form["type"]

    # Convert type into dummy variables
    type_CASH_OUT = 0
    type_DEBIT=0
    type_PAYMENT = 0
    type_TRANSFER = 0

    if txn_type == "CASH_OUT":
        type_CASH_OUT = 1
    elif txn_type == "DEBIT":
        type_DEBIT = 1
    elif txn_type == "PAYMENT":
        type_PAYMENT = 1
    elif txn_type == "TRANSFER":
        type_TRANSFER = 1

    final_features = [[
        step,
        amount,
        oldbalanceOrg,
        newbalanceOrig,
        oldbalanceDest,
        newbalanceDest,
        type_CASH_OUT,
        type_DEBIT,
        type_PAYMENT,
        type_TRANSFER
    ]]

    prediction = model.predict(final_features)

    if prediction[0] == "is Fraud":
        result = "⚠️ Fraud Transaction Detected!"
    else:
        result = "✅ Legitimate Transaction"

    return render_template("submit.html", prediction_text=result)


if __name__ == "__main__":
    app.run(debug=True)

