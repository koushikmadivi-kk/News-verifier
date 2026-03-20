from flask import Flask, render_template, request
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    text = ""

    if request.method == "POST":
        text = request.form["text"]
        data = [text]
        vect = vectorizer.transform(data)
        result = model.predict(vect)
        
        if result[0] == "REAL":
            prediction = "This news article is likely REAL."
        else:
            prediction = "This news article is likely FAKE."

    return render_template("index.html", prediction=prediction, text=text)

if __name__ == "__main__":
    app.run(debug=True)
