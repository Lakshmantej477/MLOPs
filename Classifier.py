import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("/content/MLOPs/data/iris.csv") 
features = ['sepal.length','sepal.width','petal.length','petal.width']
target = 'variety'

x_train,x_test,y_train,y_test = train_test_split(df[features],df[target])
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

print(f'accuracy of the model is {accuracy_score(y_test,y_pred)*100}')