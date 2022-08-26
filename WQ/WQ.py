# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.model_selection import (
    KFold,
    train_test_split,
    GridSearchCV,
)
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from sklearn.metrics import r2_score


# -

# CARGANDO EL CONJUNTO DE DATOS

dfwater = pd.read_csv('water_potability.csv')
dfwater.info()

# DIOMENSIONES DEL DATASET

dfwater.shape

# REVISANDO EL CONTENIDO DEL DATASET

# +

dfwater.head(20)
# -

dfwater.describe()

dfwater['Potability'].value_counts()

# GRAFICANDO LOS DATOS

ax=plt.subplots(1,1,figsize=(10,8))
sns.countplot(x='Potability',data=dfwater,palette="Set2")
plt.title("Potabilidad")
plt.show()

# BUSCANDO REGISTROS CON VALORES VACIOS

dfwater.isna().sum().sort_values()

# CORRGIENDO LOS REGISTROS CON VALORES VACIOS

dfwater['Trihalomethanes'].fillna(value=dfwater['Trihalomethanes'].median(), inplace=True)
dfwater['ph'].fillna(value=dfwater['ph'].median(), inplace=True)
dfwater['Sulfate'].fillna(value=dfwater['Sulfate'].median(), inplace=True)

dfwater.head(20)

dfwater.isnull().sum()

# NIVELANDO LOS DATOS

y_target = dfwater['Potability']
x_data = dfwater.drop(['Potability'], axis=1)
smt = SMOTE(random_state=42)
x_data, y_target = smt.fit_resample(x_data, y_target)
sns.countplot(y_target)

# ESCALANDO LOS DATOS

scaler = MinMaxScaler()
x_data = pd.DataFrame(scaler.fit_transform(x_data), columns =x_data.columns.to_list())

# DIVIENDO EL CONJUNTO DE DATOS PARA ENTRENAMIENTO Y PRUEBA

x_train, x_test, y_train, y_test = train_test_split(x_data, y_target, test_size=0.2)

# APLICANDO ARBOLES DE DECISION

# +
# Decision Tree Classifier
dtc = DecisionTreeClassifier()

# hyperparameter space
param_grid = dict(
    max_depth = [25,30,35],
    min_samples_split = [4,5,6,7]
)

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=4)

# search
clf =  GridSearchCV(
    dtc,
    param_grid,
    scoring='accuracy',
    cv=kf, # k-fold
    refit=True, # refits best model to entire dataset
)

search = clf.fit(x_train, y_train)


# best hyperparameters
search.best_params_
# -

results = pd.DataFrame(search.cv_results_)[['params', 'mean_test_score', 'std_test_score']]
print(results.shape)
results

# MOSTRANDO RESULTADOS Y MATRIZ DE CONFUSION

# +
# let's get the predictions
train_preds = search.predict(x_train)
test_preds = search.predict(x_test)

print('Train Accuracy: ', accuracy_score(y_train, train_preds))
print('Test Accuracy: ', accuracy_score(y_test, test_preds))

print("Confusion matrix:\n%s" % confusion_matrix(y_test, test_preds))

# -

new_data = pd.read_csv('datos_nuevos.csv')
new_data.shape
new_data.head()

# +
new_preds = search.predict(new_data)
print(new_preds)



# -


