import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

application = Flask(
    __name__, template_folder="templates", static_folder="static"
)  # Initialize the flask App
model = pickle.load(open("model.pkl", "rb"))


@application.route("/")
def home():
    return render_template("index.html")  # The form for user input


@application.route("/predict", methods=["POST"])
def predict():
    area = float(request.form["area"])
    bedrooms = float(request.form["bedrooms"])
    bathrooms = float(request.form["bathrooms"])
    stories = float(request.form["stories"])
    mainroad = float(request.form["mainroad"])

    prediction = model.predict([[area, bedrooms, bathrooms,stories,mainroad]])

    # Code to generate the regression line graph goes here

    return render_template("result.html", price_prediction=round(prediction[0],2))


if __name__ == "__main__":
    application.run(debug=True)
