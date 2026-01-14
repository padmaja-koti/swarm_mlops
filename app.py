from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/", methods=["GET"])
def home():
    return "Iris ML API is running. Use POST /predict"

@app.route("/predict", methods=["POST"])
def predict():
    features = request.json["features"]
    prediction = model.predict([features])[0]
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)