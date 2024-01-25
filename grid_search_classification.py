import itertools
import typing
import json
import numpy as np
from sklearn import datasets
from tabular_data import load_airbnd
from tabular_data import clean_tabular_data
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
import joblib
from sklearn.linear_model import LogisticRegression
features, labels=load_airbnd(df=clean_tabular_data())
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score
import os
X_train, X_validation, y_train, y_validation = train_test_split(features, labels, test_size=0.2)

def grid_search(hyperparameters: typing.Dict[str, typing.Iterable]):
    keys, values = zip(*hyperparameters.items())
    yield from (dict(zip(keys, v)) for v in itertools.product(*values))
grid = {
    "validation_accuracy": [ "accuracy_score"]
    }
for i, hyperparams in enumerate(grid_search(grid)):
    print(i, hyperparams)
def k_fold(dataset, n_splits: int = 5):
    chunks = np.array_split(dataset, n_splits)
    for i in range(n_splits):
        training = chunks[:i] + chunks[i + 1 :]
        validation = chunks[i]
        yield np.concatenate(training), validation

model=LogisticRegression()
def tune_classification_model_hyperparameters(model, X_train, X_validation, y_train, y_validation,hyperparams):
    
    # K-Fold evaluation
    best_hyperparams, best_loss = None, np.inf
    n_splits = 5
    # Grid search goes first
    for hyperparams in grid_search(grid):
        loss = 0
        # Instead of validation we use K-Fold
        for (X_train, X_validation), (y_train, y_validation) in zip(
            k_fold(features, n_splits), k_fold(labels, n_splits)
        ):
            model=LogisticRegression()
            model.fit(X_train, y_train)

            y_validation_pred = model.predict(X_validation)
            fold_loss =accuracy_score(y_validation, y_validation_pred)
            loss += fold_loss
        # Take the mean of all the folds as the final validation score
        total_loss = loss / n_splits
        print(f"H-Params: {hyperparams} Loss: {total_loss}")
        if total_loss < best_loss:
            best_loss = total_loss
            best_hyperparams = hyperparams

    # See the final results
    print(f"Best loss: {best_loss}")
    print(f"Best hyperparameters: {best_hyperparams}")
    return (1-best_loss), best_hyperparams
print(tune_classification_model_hyperparameters(model, X_train, X_validation, y_train, y_validation,hyperparams))

tune_classification_model_hyperparameters(model, X_train, X_validation, y_train, y_validation,hyperparams)[1]

#save the models
dirname = os.path.dirname(__file__)
def save_models(folder):
    joblib.dump(model, folder+"model.joblib")

    with open(folder+"hyperparameters.json", 'w') as f:
        json.dump(tune_classification_model_hyperparameters(model, X_train, X_validation, y_train, y_validation,hyperparams)[1], f)

    with open(folder+"metrics.json", 'w') as f1:
        json.dump(tune_classification_model_hyperparameters(model, X_train, X_validation, y_train, y_validation,hyperparams)[0], f1)
folder= r"C:/Users/laura/OneDrive/Desktop/Data Science/models/classification/logistic_regression"
save_models(folder)