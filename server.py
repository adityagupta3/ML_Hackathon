# Dependencies
from flask import Flask, request, jsonify
import traceback
import joblib
import pandas as pd
import numpy as np
#import model_ML
from joblib import dump, load

def predict(classifier):
    if ( classifier == lightgbm_classifier):
        classifier = lightgbm_classifier
        model = "lightgbm_classifier"
    elif ( classifier == rf_classifier):
        classifier = rf_classifier
        model = "rf_classifier"
    elif ( classifier == xgboost_classifier):
        classifier = xgboost_classifier
        model = "xgboost_classifier"
    else:
        print ('Train the model first')
        return ('No model here to use')
    try: 
        json_ = request.json
        #print(json_)
        query = pd.DataFrame(json_)
        #print(query)
        X = query
        #print(X.iloc[:,0:5])
        #print(encoder_dict)
        label = X.iloc[:,0:5].apply( lambda x: encoder_dict[x.name].transform(x))
        #print(label)
        label['amount'] = X.iloc[:,5]
        X = label.values
        #print(X)
        X[:,[5]] = sc.transform(X[:,[5]])
        #print(X)
        prediction = classifier.predict(X).round(0)
        #print(prediction)
        result =0
        if prediction==0:
            result = "GENUINE TRANSACTION"
        elif prediction==1:
            result = "FRAUD TRANSACTION"
            
        return jsonify({'PREDICTION': str(prediction),'RESULT': str(result),'MODEL USED': model})

    except:

        return jsonify({'trace': traceback.format_exc()})
    



# Your API definition
app = Flask(__name__)

@app.route('/predict_lightgbm', methods=['POST'])
def predict_lightgbm():
    json = predict(lightgbm_classifier)
    return json

@app.route('/predict_xgboost', methods=['POST'])
def predict_xgboost():
    predict(xgboost_classifier)
    json = predict(xgboost_classifier)
    return json

@app.route('/predict_rf', methods=['POST'])
def predict_rf():
    predict(rf_classifier)
    json = predict(rf_classifier)
    return json



if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 8080 

    # Load "model.pkl" LIGHTGBM
    lightgbm_classifier = joblib.load("modules\\lightgbm_model.pkl") 
    xgboost_classifier = joblib.load("modules\\xgboost_model.pkl")
    rf_classifier = joblib.load("modules\\rf_model.pkl") 
    print ('Model loaded')

    # Importing Objects used to encode models from saved files
    #encoder_dict and std_scaler
    sc=load('modules\\std_scaler.bin')
    encoder_dict = load('modules\\label_encoder.bin')

    app.run(port=port, debug=True)