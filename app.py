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
    
@app.route('/predictFasttext', methods=['POST'])
def predictFasttext():
    # queries = json.loads(request.form)
    model = fasttext.load_model('fasttext_train1.bin')
    input_json = request.json
    queryString = input_json['query'];
    predict = model.predict(queryString)
    return jsonify({"query":queryString, "group": str(predict[0][0]) })

@app.route('/predictFasttexttop5', methods=['POST'])
def predictFasttexttop5():
    model = fasttext.load_model('fasttext_train_top5.bin')
    input_json = request.json
    queryString = input_json['query'];
    predict = model.predict(queryString)
    #return jsonify({"query":queryString, "group": str(predict[0][0]) })
    return jsonify({"query":queryString, "group": "Inside fasttext_top5" })

@app.route('/LSTM', methods=['POST'])
def predictLSTM():
    model = fasttext.load_model('fasttext_train1.bin')
    input_json = request.json
    queryString = input_json['query'];
    predict = model.predict(queryString)
    #return jsonify({"query":queryString, "group": str(predict[0][0]) })
    return jsonify({"query":queryString, "group": "Inside LSTM" })

@app.route('/predictBI_LSTM', methods=['POST'])
def predictBI_LSTM():
    model = fasttext.load_model('fasttext_train1.bin')
    input_json = request.json
    queryString = input_json['query'];
    predict = model.predict(queryString)
    #return jsonify({"query":queryString, "group": str(predict[0][0]) })
    return jsonify({"query":queryString, "group": "Inside Bi-directonal LSTM" })



if __name__ == '__main__': app.run(debug=True)