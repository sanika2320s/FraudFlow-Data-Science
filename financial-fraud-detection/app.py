from flask import Flask, render_template, request, url_for
import os

# try to load a real model if present
MODEL_PATH = os.path.join("models", "fraud_pipeline.joblib")
use_model = False
model = None
try:
    import joblib
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        use_model = True
except Exception:
    use_model = False

app = Flask(__name__)

def predict_rule_based(cibil, fraud_info, income_source, tax_paid, loan_amount, loan_interest, timely_paid):
    """
    Simple heuristic fallback:
    returns probability-like score (0-1) and label (0 = No Fraud, 1 = Fraud)
    """
    score = 0.0
    # suspicious fraud flag
    if fraud_info == 1:
        score += 0.6
    # low credit score adds risk
    if cibil < 600:
        score += 0.25
    elif cibil < 700:
        score += 0.1
    # large loans + high interest increases risk
    if loan_amount and loan_amount > 300000:
        score += 0.15
        if timely_paid == "No":
            score += 0.2
    # tax not paid and non-salary source adds small risk
    if tax_paid == "No":
        score += 0.08
    if income_source.lower() not in ("salary", "investments"):
        score += 0.05
    # clamp
    score = min(1.0, score)
    label = 1 if score >= 0.5 else 0
    return score, label

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/fraud_form")
def fraud_form():
    return render_template("fraud_form.html")

@app.route("/loan_details")
def loan_details():
    return render_template("loan_details.html")

@app.route("/predict", methods=["POST"])
def predict():
    # parse inputs safely
    def to_float(val):
        try:
            return float(val)
        except Exception:
            return 0.0

    try:
        cibil = to_float(request.form.get("cibil", 0))
        fraud_info = int(request.form.get("fraud_info", 0))
    except Exception:
        cibil = 0.0
        fraud_info = 0

    income_source = request.form.get("income_source", "Salary")
    tax_paid = request.form.get("tax_paid", "Yes")
    timely_paid = request.form.get("timely_paid", "Yes")

    loan_amount = to_float(request.form.get("loan_amount", 0))
    loan_interest = to_float(request.form.get("loan_interest", 0))

    # if there is a real model, try to use it
    if use_model and model is not None:
        try:
            # NOTE: adapt this to your model's expected features & preprocessing
            features = [[cibil, fraud_info, loan_amount, loan_interest]]
            y_pred = model.predict(features)
            # If model returns class label:
            label = int(y_pred[0])
            # If model returns probabilities:
            try:
                prob = model.predict_proba(features)[0][1]
            except Exception:
                prob = 0.9 if label == 1 else 0.1
        except Exception:
            prob, label = predict_rule_based(cibil, fraud_info, income_source, tax_paid, loan_amount, loan_interest, timely_paid)
    else:
        prob, label = predict_rule_based(cibil, fraud_info, income_source, tax_paid, loan_amount, loan_interest, timely_paid)

    # format probability percentage
    prob_pct = int(round(prob * 100))

    return render_template(
        "fraud_result.html",
        cibil=cibil,
        fraud_info=fraud_info,
        income_source=income_source,
        tax_paid=tax_paid,
        timely_paid=timely_paid,
        loan_amount=loan_amount,
        loan_interest=loan_interest,
        prob=prob_pct,
        label=label
    )

if __name__ == "__main__":
    app.run(debug=True)
