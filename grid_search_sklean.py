from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import itertools
import typing
import joblib
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import os
import json


X, y = datasets.fetch_california_housing(return_X_y=True)


def grid_search(hyperparameters: typing.Dict[str, typing.Iterable]):
    keys, values = zip(*hyperparameters.items())
    yield from (dict(zip(keys, v)) for v in itertools.product(*values))
grid = {
    "criterion": ["mse", "squared_error"],
    "min_samples_leaf": [2, 1]
}

def evaluate_all_models():
    for i, hyperparams in enumerate(grid_search(grid)):
        print(i, hyperparams)

    modelRF = RandomForestRegressor(**hyperparams)
    X, y = datasets.fetch_california_housing(return_X_y=True)

    def tune_RFregression_model_hyperparameters():
        GS = GridSearchCV(estimator=modelRF,param_grid=grid, scoring='r2', refit='r2', cv=5)
        print(GS.fit(X, y))
        print(GS.best_params_)
        print(GS.best_score_)
        return (GS.fit(X, y), GS.best_params_,GS.best_score_)
    # print(tune_RFregression_model_hyperparameters())


    modelDT = DecisionTreeRegressor(criterion='squared_error', splitter='best', max_depth=None, min_samples_split=2)


    def tune_DTregression_model_hyperparameters():
        GS = GridSearchCV(estimator=modelDT,param_grid=grid, scoring='r2', refit='r2', cv=5)
        print(GS.fit(X, y))
        print(GS.best_params_)
        print(GS.best_score_)
        return (GS.fit(X, y), GS.best_params_,GS.best_score_)
    # print(tune_DTregression_model_hyperparameters())

    modelGB = GradientBoostingRegressor(loss='squared_error', learning_rate=0.1, n_estimators=100)


    def tune_GBregression_model_hyperparameters():
        GS = GridSearchCV(estimator=modelGB,param_grid=grid, scoring='r2', refit='r2', cv=5)
        print(GS.fit(X, y))
        print(GS.best_params_)
        print(GS.best_score_)
        return (GS.fit(X, y), GS.best_params_,GS.best_score_)
    # print(tune_GBregression_model_hyperparameters())
    return(tune_DTregression_model_hyperparameters(), tune_RFregression_model_hyperparameters(),tune_GBregression_model_hyperparameters())

# Save the models 

    dirname = os.path.dirname(__file__)
    def save_models_RF(folder):
        joblib.dump(tune_RFregression_model_hyperparameters()[0], folder+"model.joblib")

        with open(folder+"hyperparameters.json", 'w') as f:
            json.dump(tune_RFregression_model_hyperparameters()[1], f)

        with open(folder+"metrics.json", 'w') as f1:
            json.dump(tune_RFregression_model_hyperparameters()[2], f1)


    folder= r"C:/Users/laura/OneDrive/Desktop/Data Science/models/regression/Random_Forest/"
    save_models_RF(folder)




    def save_models_DT(folder):
        joblib.dump(tune_DTregression_model_hyperparameters()[0], folder+"model.joblib")

        with open(folder+"hyperparameters.json", 'w') as f:
            json.dump(tune_DTregression_model_hyperparameters()[1], f)

        with open(folder+"metrics.json", 'w') as f1:
            json.dump(tune_DTregression_model_hyperparameters()[2], f1)


    folder= r"C:/Users/laura/OneDrive/Desktop/Data Science/models/regression/Decision_Tree/"
    save_models_DT(folder)



    def save_models_GB(folder):
        joblib.dump(tune_GBregression_model_hyperparameters()[0], folder+"model.joblib")

        with open(folder+"hyperparameters.json", 'w') as f:
            json.dump(tune_GBregression_model_hyperparameters()[1], f)

        with open(folder+"metrics.json", 'w') as f1:
            json.dump(tune_GBregression_model_hyperparameters()[2], f1)


    folder= r"C:/Users/laura/OneDrive/Desktop/Data Science/models/regression/Gradient_Boosting/"
    save_models_GB(folder)

def find_best_model():
    result=pd.DataFrame(evaluate_all_models())
    ind=np.where(result[2]==max(result[2]))[0]
    # print(result[0][ind],result[1][ind],result[2][ind])
    return (result[0][ind],result[1][ind],result[2][ind])

if __name__ == "__main__":
    evaluate_all_models()
    find_best_model()