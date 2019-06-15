
from flask import Flask, request
from predict import predict_price, build_model
app = Flask(__name__)

build_model()

@app.route("/pricerange", methods = ['POST'])
def rec_response():
    content = request.json
    reviews = content["reviews"]
    pricerange = predict_price(reviews)
    return price_range

@app.route("/")
def hello():
    return "Hello World!"

if __name__ == '__main__':
    app.run(host= '0.0.0.0',port=7777)
    