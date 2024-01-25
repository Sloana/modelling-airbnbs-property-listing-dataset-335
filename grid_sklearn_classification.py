

import itertools
import typing
import json
import numpy as np
from sklearn import datasets
from tabular_data import load_airbnd
from tabular_data import clean_tabular_data
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
X, y=load_airbnd(df=clean_tabular_data())
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score
import os
import pandas as pd
def grid_search(hyperparameters: typing.Dict[str, typing.Iterable]):
    keys, values = zip(*hyperparameters.items())
    yield from (dict(zip(keys, v)) for v in itertools.product(*values))
def grid_search(hyperparameters: typing.Dict[str, typing.Iterable]):
    keys, values = zip(*hyperparameters.items())
    yield from (dict(zip(keys, v)) for v in itertools.product(*values))
grid = {
    "min_samples_leaf": [2, 1]
}
grid_score = {
    "validation_accuracy": [ "accuracy_score"]
    }
def evaluate_all_models():

    modelRF = RandomForestClassifier(max_depth=2, random_state=0)

    def tune_RFclassification_model_hyperparameters():
        GS = GridSearchCV(estimator=modelRF,param_grid=grid, cv=5)
        print(GS.fit(X, y))
        print(GS.best_params_)
        print(GS.best_score_)
        return (GS.fit(X, y), GS.best_params_,GS.best_score_)
    # print(tune_RFregression_model_hyperparameters())


    modelDT = DecisionTreeClassifier(max_depth=2,random_state=0)


    def tune_DTclassification_model_hyperparameters():
        GS = GridSearchCV(estimator=modelDT,param_grid=grid,cv=5)
        print(GS.fit(X, y))
        print(GS.best_params_)
        print(GS.best_score_)
        return (GS.fit(X, y), GS.best_params_,GS.best_score_)
    # print(tune_DTclassification_model_hyperparameters())

    modelGB = GradientBoostingClassifier(max_depth=2,random_state=0)


    def tune_GBclassification_model_hyperparameters():
        GS = GridSearchCV(estimator=modelGB, param_grid=grid, cv=5)
        print(GS.fit(X, y))
        print(GS.best_params_)
        print(GS.best_score_)
        return (GS.fit(X, y), GS.best_params_,GS.best_score_)
    # print(tune_GBclassification_model_hyperparameters())
    return(tune_DTclassification_model_hyperparameters(), tune_RFclassification_model_hyperparameters(),tune_GBclassification_model_hyperparameters())

# Save the models 

    dirname = os.path.dirname(__file__)
    def save_models_RF(folder):
        joblib.dump(tune_RFclassification_model_hyperparameters()[0], folder+"model.joblib")

        with open(folder+"hyperparameters.json", 'w') as f:
            json.dump(tune_RFclassification_model_hyperparameters()[1], f)

        with open(folder+"metrics.json", 'w') as f1:
            json.dump(tune_RFclassification_model_hyperparameters()[2], f1)


    folder= r"C:/Users/laura/OneDrive/Desktop/Data Science/models/classification/Random Forest/"
    save_models_RF(folder)




    def save_models_DT(folder):
        joblib.dump(tune_DTclassification_model_hyperparameters()[0], folder+"model.joblib")

        with open(folder+"hyperparameters.json", 'w') as f:
            json.dump(tune_DTclassification_model_hyperparameters()[1], f)

        with open(folder+"metrics.json", 'w') as f1:
            json.dump(tune_DTclassification_model_hyperparameters()[2], f1)


    folder= r"C:/Users/laura/OneDrive/Desktop/Data Science/models/classification/Decision Tree/"
    save_models_DT(folder)



    def save_models_GB(folder):
        joblib.dump(tune_GBclassification_model_hyperparameters()[0], folder+"model.joblib")

        with open(folder+"hyperparameters.json", 'w') as f:
            json.dump(tune_GBclassification_model_hyperparameters()[1], f)

        with open(folder+"metrics.json", 'w') as f1:
            json.dump(tune_GBclassification_model_hyperparameters()[2], f1)


    folder= r"C:/Users/laura/OneDrive/Desktop/Data Science/models/classification/Gradient_Boosting/"
    save_models_GB(folder)

def find_best_model():
    result=pd.DataFrame(evaluate_all_models())
    ind=np.where(result[2]==max(result[2]))[0]
    # print(result[0][ind],result[1][ind],result[2][ind])
    return (result[0][ind],result[1][ind],result[2][ind])

if __name__ == "__main__":
    evaluate_all_models()
    find_best_model()