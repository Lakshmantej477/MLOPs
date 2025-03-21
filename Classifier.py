import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# df = pd.read_csv("/content/MLOPs/data/iris.csv") 
# df = pd.read_csv("C:\Users\admin\Downloads\new_MLOPs\data\iris.csv")


# azureml-core of version 1.0.72 or higher is required
# azureml-dataprep[pandas] of version 1.1.34 or higher is required
from azureml.core import Workspace, Dataset

subscription_id = 'a99817d4-8cc7-42ee-abdf-5801f4f736ba'
resource_group = 'ilakshmantej-rg'
workspace_name = 'Lakshmantej'

workspace = Workspace(subscription_id, resource_group, workspace_name)

dataset = Dataset.get_by_name(workspace, name='Iris')
df = dataset.to_pandas_dataframe()



features = ['sepal.length','sepal.width','petal.length','petal.width']
target = 'variety'

x_train,x_test,y_train,y_test = train_test_split(df[features],df[target],test_size=0.2)
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

print(f'accuracy of the model is {accuracy_score(y_test,y_pred)*100}')
