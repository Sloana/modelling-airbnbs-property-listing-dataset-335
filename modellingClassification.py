import pandas as pd
import math
from tabular_data import clean_tabular_data
from tabular_data import load_airbnd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score


import itertools
import typing
df = pd.read_csv(r"C:/Users/laura/OneDrive/Desktop/Data Science/airbnb-property-listings/tabular_data/listing.csv")
model=LogisticRegression(random_state=0)
features, labels=load_airbnd(df=clean_tabular_data())
features = features.apply( pd.to_numeric, errors='coerce' )

features[["guests"]] = features[["guests"]].astype(float)
print(labels.dtypes)
print(features.dtypes)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
# Calculate the performance for your classification model. That should include the F1 score, the precision, the recall, and the accuracy for both the training and test sets.
rcall_train =recall_score(y_train, y_train_pred, average=None)
rcall_test= recall_score(y_test, y_test_pred, average=None)
f1_train=f1_score(y_train, y_train_pred, average=None)
f1_test=f1_score(y_test, y_test_pred, average=None)
precision_train =precision_score(y_train, y_train_pred, average=None)
precision_test= precision_score(y_test, y_test_pred, average=None)
accuracy_train=accuracy_score(y_train, y_train_pred)
accuracy_test=accuracy_score(y_test, y_test_pred)
print(
f"{model.__class__.__name__}: "
f"rcall_Train: {rcall_train} | "
f"rcall_Test: {rcall_test}"
)


print(
f"{model.__class__.__name__}: "
f"f1_Train: {f1_train} | "
f"f1_Test: {f1_test}"
)


print(
f"{model.__class__.__name__}: "
f"precision_Train: {precision_train} | "
f"precision_Test: {precision_test}"
)


print(
f"{model.__class__.__name__}: "
f"accuracy_Train: {accuracy_train} | "
f"accuracy_Test: {accuracy_test}"
)
