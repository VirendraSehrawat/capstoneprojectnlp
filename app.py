from flask import Flask, render_template
from flask import request
import fasttext
from flask.json import jsonify
import json

app = Flask(__name__, static_folder='templates')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['GET'])
def getCategory():
    model = fasttext.load_model('fasttext_train1.bin')
    try:
        input_string = request.args['query']
        predict = model.predict(input_string)
        action = {"Description":input_string,
        "Suggested Group":str(predict[0])[11:-2]
        }
        return str(action)
    except :
        return str("Error reading query")
    
@app.route('/predictPost', methods=['POST'])
def predictPost():
    # queries = json.loads(request.form)
    model = fasttext.load_model('fasttext_train1.bin')
    input_json = request.json
    queryString = input_json['query'];
    predict = model.predict(queryString)
    return jsonify({"query":queryString, "group": str(predict[0][0]) })

@app.route('/predictBert', methods=['POST'])
def predictBert():
    input_json = request.json
    queryString = input_json['query'];
    return jsonify({"query":queryString, "group": "add bert predictions" })

@app.route('/predictDirt', methods=['POST'])
def predictDirt():
    input_json = request.json
    queryString = input_json['query'];
    return jsonify({"query":queryString, "group": "add dirt predictions" })

@app.route('/PredictGotIt', methods=['POST'])
def PredictGotIt():
    input_json = request.json
    queryString = input_json['query'];
    return jsonify({"query":queryString, "group": "add got it predictions" })

if __name__ == '__main__': app.run(debug=True)