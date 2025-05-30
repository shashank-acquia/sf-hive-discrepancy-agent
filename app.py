from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from main import dw_validation,getColumnList,getConvertedScript
import json

load_dotenv()

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    name = request.form["name"]
    
    suggestions, discrepancy_json , expanded_scr_map = dw_validation(name=name)

    return jsonify({
        "suggestions": suggestions,
        "discrepancies": discrepancy_json,
        "expanded_scr_map":expanded_scr_map
    })


@app.route("/convert", methods=["POST"])
def convert():
    script = request.form["script"]
    
    suggestions = getConvertedScript(script=script)
    data = json.loads(suggestions)

    return jsonify({
        "converted_script": data['results'],
    })
@app.route("/metadata", methods=["GET"])
def metadata():
    col_list = getColumnList()

    return jsonify({
        "col_list": col_list
    })


if __name__ == "__main__":

    app.run(host="0.0.0.0", debug=True)