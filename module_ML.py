# Import dependencies
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from collections import defaultdict
from joblib import dump, load

def data_processing(csv,train,model_name):
    # Load the dataset in a dataframe object and include only four features as mentioned
    dataset=pd.read_csv(csv)
    #dataset=pd.read_csv("fraud_final.csv")
    dataset= dataset.drop(['zipcodeOri','zipMerchant','step'] , axis=1)

    lists = ['E','U']
    for item in lists:
        index_names = dataset[ dataset['gender'] == item].index
        dataset = dataset.drop(index_names)

    # Data Preprocessing
    sc = StandardScaler()
    encoder_dict = defaultdict(LabelEncoder)

    Y=dataset.iloc[:,-1].values
    dataset.iloc[:,0:5].apply(lambda x: encoder_dict[x.name].fit(x))
    labeled = dataset.iloc[:,0:5].apply(lambda x: encoder_dict[x.name].transform(x))
    labeled['amount'] = dataset.iloc[:,5]
    X = labeled.values
    sc.fit(X[:,[5]])
    X[:,[5]] = sc.transform(X[:,[5]])

    #def encoder_dictonary():
    dump(encoder_dict, 'modules\\label_encoder.bin', compress=True)

    #def gen_StandardScaler():
    dump(sc, 'modules\\std_scaler.bin', compress=True)

    model_columns = dataset.columns
    print("Columns = "+ str(model_columns))
    joblib.dump(model_columns, 'modules\\model_columns.pkl')
    print("Models columns dumped!")

    dataset= dataset.drop(['fraud'] , axis=1)

    if(train==1):
        print("Training the model.....")
        model(X,Y,model_name)
    else:
        print("Model not trained , Using saved module")
    


def model(X,Y,model_name):
    #Train and Test Data
    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,random_state=42)

    if (str(model_name).lower() == "lightgbm" ):
        import lightgbm as lgb
        train_data=lgb.Dataset(X_train,label=y_train)
        param = {'num_leaves':100, 'objective':'binary','max_depth':7,'learning_rate':.05,'max_bin':500}
        param['metric'] = ['auc', 'binary_logloss']
        num_round=500
        lgbm=lgb.train(param,train_data,num_round)
        y_pred=lgbm.predict(X_test)
        #rounding the values
        y_pred=y_pred.round(0)
        #converting from float to integer
        y_pred=y_pred.astype(int)

        accuracy_lgbm = accuracy_score(y_test,y_pred.round())

        accuracy = accuracy_score(y_test,y_pred.round())
        print("Accuracy of LIGHTGBM= "+str(accuracy_lgbm))
        joblib.dump(lgbm, 'modules\\lightgbm_model.pkl')
        print("LIGHTGBM Model dumped!")

    if (str(model_name).lower() == "xgboost" ):
        import xgboost as xgb
        classifier = xgb.XGBClassifier(max_depth=6, learning_rate=0.05, n_estimators=400, 
                                        objective="binary:hinge", booster='gbtree', 
                                        n_jobs=-1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, 
                                        subsample=1, colsample_bytree=0.05, colsample_bylevel=0.05, reg_alpha=0, reg_lambda=1, 
                                        scale_pos_weight=1, base_score=0.5, random_state=42)

        classifier.fit(X_train,y_train)
        y_pred = classifier.predict(X_test)

        accuracy_xgboost = accuracy_score(y_test,y_pred.round())
        print(accuracy_xgboost)

        accuracy = accuracy_score(y_test,y_pred.round())
        print("Accuracy of XGBOOST= "+str(accuracy_xgboost))
        joblib.dump(classifier, 'modules\\xgboost_model.pkl')
        print("XGBOOST Model dumped!")

    if (str(model_name).lower() == "randomforest" ):
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        accuracy_rf = accuracy_score(y_test,y_pred.round())
        print(accuracy_rf)

        accuracy = accuracy_score(y_test,y_pred.round())
        print("Accuracy of Random forest = "+str(accuracy_rf))
        joblib.dump(classifier, 'modules\\rf_model.pkl')
        print("Random Forest Model dumped!")