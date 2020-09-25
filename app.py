from flask import Flask, render_template
from flask import request
import fasttext
from flask.json import jsonify
import json
# import tensorflow as tf

app = Flask(__name__, static_folder='templates')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/getPredictMethods', methods = ['GET'])
def getMethods():
    return jsonify([
                    {"displayname": "Fast Text All Groups", "method": "predictFasttext"},
                    {"displayname": "Fast Text Top 5 Groups", "method": "predictFasttexttop5"},
                    {"displayname": "Fast Text Top 10 Groups", "method": "predictFasttexttop10"},
                    {"displayname": "FT T5 with Preprocessing and without Lemmatization", "method": "predictFtPNl"},
                    {"displayname": "FT T5 without Preprocessing and Lemmatization", "method": "predictFtNpNl"}
                    ]);

@app.route('/predictFasttext', methods=['POST'])
def predictFasttext():    
    return predict('Fasttext_Allgroups.bin', 3)

@app.route('/predictFasttexttop5', methods=['POST'])
def predictFasttexttop5():    
    return predict('Fasttext_Top5Groups.bin', 2)

@app.route('/predictFasttexttop10', methods=['POST'])
def predictFasttexttop10():    
    return predict('Fasttext_Top10Groups.bin', 2)

@app.route('/predictFtNpNl', methods=['POST'])
def predictFasttextNpNl():    
    return predict('Fasttext_Top5Groups_NoPreprocess_NoLemm.bin', 2)

@app.route('/predictFtPNl', methods=['POST'])
def predictFasttextpNl():    
    return predict('Fasttext_Top5Groups_Preprocess_NoLemm.bin', 2)

def predict(modelName, count):
    model = fasttext.load_model(modelName)
    input_json = request.json
    queryString = input_json['query'];
    prediction = model.predict(queryString, k = count);
    print('prediction is ', prediction)
    return jsonifyPrediction(prediction, queryString)

def jsonifyPrediction(prediction, queryString):
    jsonPrediction = json.dumps(getGroupAndProbabilites(prediction))
    return jsonify({"query":queryString, "group": str(prediction[0][0].replace('__label__','')), "additionalData" : jsonPrediction})

def getGroupAndProbabilites(predictions):
    objectList = []
    for index in range(len(predictions[0])): 
        objectList.append( {"group": predictions[0][index].replace('__label__',''), "probability": predictions[1][index]})
    return objectList



if __name__ == '__main__': app.run(debug=True)