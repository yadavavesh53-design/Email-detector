from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    email = request.form["email"]
    email_vector = vectorizer.transform([email])
    prediction = model.predict(email_vector)[0]

    if prediction == "spam":
        result = "Spam Email"
    else:
        result = "Safe Email"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)