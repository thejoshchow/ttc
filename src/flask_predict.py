from flask import Flask, jsonify
from joblib import load
import pandas as pd

app = Flask(__name__)
@app.route('/predict', methods=['POST'])

def predict():
     json_ = request.json
     query_df = pd.DataFrame(json_)
     query = pd.get_dummies(query_df)
     prediction = to_4.predict(from_3.predict(query))
     return jsonify({'prediction': list(prediction)})

if __name__ == '__main__':
     from_3 = load('from_3.pkl')
     to_4 = load('to_4.pkl')
     app.run(port=8080)