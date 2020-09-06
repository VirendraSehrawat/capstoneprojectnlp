from flask import Flask, render_template
from flask import request

app = Flask(__name__, static_folder='templates')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['GET'])
def getCategory():
    try:
        return  str("add model response for --"+request.args['query']) #//str(data)
    except :
        return str("Error reading query")
    
    




if __name__ == '__main__': app.run(debug=True)