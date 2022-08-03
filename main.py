import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

insurance = pd.read_csv('insurance.csv')
print(insurance.head())
print(insurance.info())
print(insurance.isnull().sum())  # no missing values in the dataset
lab_enc = LabelEncoder()  # Labelling all non-numeric data fields into nominal numbers
insurance['sex'] = lab_enc.fit_transform(insurance['sex'])
insurance['smoker'] = lab_enc.fit_transform(insurance['smoker'])
insurance['region'] = lab_enc.fit_transform(insurance['region'])
X = insurance.drop(columns='charges')
Y = insurance['charges']


print(X)
print(Y)
model = LinearRegression()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
model.fit(X_train, Y_train)

eg1 = (61, 0, 29.070, 0, 1, 1)
eg1 = np.asarray(eg1)
eg1 = eg1.reshape(1, -1)
eg1_prediction = model.predict(eg1)
print(eg1_prediction)

