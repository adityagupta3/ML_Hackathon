Data Source : https://www.kaggle.com/ntnu-testimon/banksim1

**Project** : Detect if a payment/transaction is fraud by training models and taking the one with best accuracy

NOTE : As same encoders are being from when the model was used by creating dictionary file and then using it  to read and encode values- any new Merchants/Customers do not tell about transaction.

Project has the following - 

- **gui.py** : Run to launch GUI implementation of model. Used tkinter module to create a basic GUI that takes in parameters and returns if the transaction is **Fraud or not**  

- **module_ML.py** : Contains model in python to generate AI model files and Encoder/Standard scaler files to be used by REST and GUI

-  **server.py** : API for model - Contains 3 models to be called by their seperate api calls. Used Flask to create server at localhost:8080 

- **modules** : Folder contains 3 model files , encoder files and data in csv to beimported by the model

- **requests.py** : Just a template to test api

