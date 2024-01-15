import pandas as pd
import math
from tabular_data import clean_tabular_data
from tabular_data import load_airbnd
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import itertools
import typing
df = pd.read_csv(r"C:/Users/laura/OneDrive/Desktop/Data Science/airbnb-property-listings/tabular_data/listing.csv")
model=SGDRegressor(max_iter=1000, tol=1e-3)
features, labels=load_airbnd(df=clean_tabular_data())

# print(features[['guests']])
features = features.apply( pd.to_numeric, errors='coerce' )

features[["guests"]] = features[["guests"]].astype(float)
print(labels.dtypes)
print(features.dtypes)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)
# X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.3)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
# y_validation_pred = model.predict(X_validation)
y_test_pred = model.predict(X_test)

rsme_train =(mean_squared_error(y_train, y_train_pred))**0.5
# validation_loss = mean_squared_error(y_validation, y_validation_pred)
rsme_test= (mean_squared_error(y_test, y_test_pred))**0.5
r2_train=r2_score(y_train, y_train_pred)
r2_test=r2_score(y_test, y_test_pred)

print(
f"{model.__class__.__name__}: "
f"rsme_Train: {rsme_train} | "
f"rsme_Test: {rsme_test}"
)


print(
f"{model.__class__.__name__}: "
f"r2_Train: {r2_train} | "
f"r2_Test: {r2_test}"
)

