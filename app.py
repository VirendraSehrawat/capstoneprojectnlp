from flask import Flask, render_template
from flask import request
import fasttext
from flask.json import jsonify
import json
import tensorflow as tf

app = Flask(__name__, static_folder='templates')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/getPredictMethods', methods = ['GET'])
def getMethods():
    return jsonify([{"displayname": "LSTM", "method": "LSTM"},
                    {"displayname": "BI LSTM", "method": "predictBI_LSTM"},
                    {"displayname": "FastText", "method": "predictFasttext"},
                    {"displayname": "Fast Text Top 5", "method": "predictFasttexttop5"}
                    ]);


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


@app.route('/LSTM', methods=['POST'])
def predictLSTM():    
    
    try:
        model = tf.keras.models.load_model('model_LSTM.h5')
        input_json = request.json
        queryString = input_json['query']
        predict = model.predict(queryString)
        action = jsonify({"query":queryString, "group": str(predict[0][0])})
        return str(action)
    except AssertionError as error:
        return str(error)
     
@app.route('/predictBI_LSTM', methods=['POST'])
def predictBI_LSTM():
    model = tf.keras.models.load_model('model_BiLSTM.bin')
    input_json = request.json
    queryString = input_json['query'];
    predict = model.predict(queryString)
    return jsonify({"query":queryString, "group": str(predict[0][0]) })
    #return jsonify({"query":queryString, "group": "Inside Bi-directonal LSTM model" })

@app.route('/predictFasttext', methods=['POST'])
def predictFasttext():    
    model = fasttext.load_model('fasttext_train1.bin')
    input_json = request.json
    queryString = input_json['query'];
    predict = model.predict(queryString)
    return jsonify({"query":queryString, "group": str(predict[0][0])})

@app.route('/predictFasttexttop5', methods=['POST'])
def predictFasttexttop5():
    model = fasttext.load_model('fasttext_train_top5.bin')
    input_json = request.json
    queryString = input_json['query'];
    predict = model.predict(queryString)
    return jsonify({"query":queryString, "group": str(predict[0][0])})

if __name__ == '__main__': app.run(debug=True)