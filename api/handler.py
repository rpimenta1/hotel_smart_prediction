from flask import Flask, request, Response
import os 
import pandas as pd
import json
import pickle
from HotelSmart.HotelSmart import PredictCancellation

model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'final_model.pkl')
model = pickle.load(open(model_path, 'rb'))

app = Flask(__name__)

@app.route('/HotelSmart/predict', methods=['POST'])
def cancellation_predict():
    test_json = request.get_json()

    if test_json:
        if isinstance(test_json, dict):
            test_raw = pd.DataFrame(test_json, index=[0])
        else:
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())

        pipeline = PredictCancellation()
        df_cleaning = pipeline.convert_binary_columns(test_raw)
        df_feature = pipeline.feature_engineering(df_cleaning)
        df_preparation = pipeline.data_preparation(df_feature)

        df_predict = pipeline.get_predictions(model, df_preparation, test_raw)

        return Response(df_predict, status=200, mimetype='application/json')
    else:
        return Response('{}', status=200, mimetype='application/json')

if __name__ == "__main__":
    port = os.environ.get('PORT', 5000)
    app.run('0.0.0.0', port=port)
