
import numpy as np

import pandas as pd
import pickle

df = pd.read_csv('Social_Network_Ads.csv')

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df['Gender']=le.fit_transform(df['Gender'])

X=df[['Gender','Age','EstimatedSalary']]
Y=df['Purchased']

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(max_depth=2)
model.fit(X,Y)

prediction=model.predict(X)

from sklearn.metrics import accuracy_score
score=accuracy_score(Y,prediction)


pickle.dump(model, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(score)
