from flask import Flask, request, jsonify
from flask_cors import CORS
import controller

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Hello Flask!"

@app.route("/logistic-predict", methods=['POST'])
def logistic_predict():
    data = request.get_json()
    title = data.get('title')
    content = data.get('content')
    return jsonify(controller.logistic_predict(title, content))

@app.route("/random-forest-predict", methods=['POST'])
def random_forest_predict():
    data = request.get_json()
    title = data.get('title')
    content = data.get('content')
    return jsonify(controller.random_forest_predict(title, content))

@app.route("/lstm-predict", methods=['POST'])
def lstm_predict():
    data = request.get_json()
    title = data.get('title')
    content = data.get('content')
    return jsonify(controller.lstm_predict(title, content))

@app.route("/bilstm-predict", methods=['POST'])
def bilstm_predict():
    data = request.get_json()
    title = data.get('title')
    content = data.get('content')
    return jsonify(controller.bilstm_predict(title, content))

if __name__ == '__main__':
    app.run(
        # host='0.0.0.0',  # lắng nghe tất cả các IP (dùng cho deploy)
        port=8080,       # port bạn muốn chạy
        debug=True       # bật debug để tự reload khi code thay đổi
    )
