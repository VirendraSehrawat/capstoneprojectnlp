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
    return jsonify([
                    {"displayname": "FastText", "method": "predictFasttext"},
                    {"displayname": "Fast Text Top 5", "method": "predictFasttexttop5"},
                    {"displayname": "Fast Text Top 10", "method": "predictFasttexttop10"}
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
    model = fasttext.load_model('Fasttext_Allgroups.bin')
    input_json = request.json
    queryString = input_json['query'];
    prediction = model.predict(queryString, k = 3);
    print('prediction is ', prediction)
    jsonPrediction = json.dumps(getGroupAndProbabilites(prediction));
    return jsonify({"query":queryString, "group": str(prediction[0][0].replace('__label__','')), "additionalData" : jsonPrediction})

def getGroupAndProbabilites(predictions):
    objectList = []
    for index in range(len(predictions[0])): 
        objectList.append( {"group": predictions[0][index].replace('__label__',''), "probability": predictions[1][index]})
    return objectList


@app.route('/predictFasttexttop5', methods=['POST'])
def predictFasttexttop5():    
    model = fasttext.load_model('Fasttext_Top5Groups_NoPreprocess_NoLemm.bin')
    try:
        input_json = request.json
        queryString = input_json['query'];
        predict = model.predict(queryString, k = 2)
        jsonPrediction = json.dumps(getGroupAndProbabilites(predict));
        return jsonify({"query":queryString,
        "group": predict[0][0].replace('__label__',''),  "additionalData" : jsonPrediction
        })
    except :
        return str("Error reading query")


@app.route('/predictFasttexttop10', methods=['POST'])
def predictFasttexttop10():    
    model = fasttext.load_model('Fasttext_Top10Groups.bin')
    try:
        input_json = request.json
        queryString = input_json['query'];
        predict = model.predict(queryString, k = 2)
        jsonPrediction = json.dumps(getGroupAndProbabilites(predict));
        return jsonify({"query":queryString,
        "group": predict[0][0].replace('__label__',''),  "additionalData" : jsonPrediction
        })
    except :
        return str("Error reading query")

if __name__ == '__main__': app.run(debug=True)