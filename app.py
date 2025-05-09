from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from main import dw_validation

load_dotenv()

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    name = request.form["name"]
    
    suggestions, discrepancy_json = dw_validation(name=name)

    return jsonify({
        "suggestions": suggestions,
        "discrepancies": discrepancy_json
    })


if __name__ == "__main__":

    app.run(host="0.0.0.0", debug=True)