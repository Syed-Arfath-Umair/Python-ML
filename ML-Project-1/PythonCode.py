import numpy as np
import pandas as pd
iris=pd.read_csv("")
x=iris.drop(columns=['class'])
y=iris['class']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
from sklearn import neighbors
classifier=neighbors.KNeighborsClassifier()
classifier.fit(x_train,y_train)
predictions=classifier.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions)*100)
from sklearn import tree
dstreeclassifier=tree.DecisionTreeClassifier()
dstreeclassifier.fit(x_train,y_train)
dstreepredictions=dstreeclassifier.predict(x_test)
print(accuracy_score(y_test,dstreepredictions)*100)
import matplotlib.pyplot as plt
fig=plt.figure()
fig.suptitle("Algorithm Comparision")
names=['KNN','Decision Tree']
result=[96.66,93.33]
plt.bar(names,result)
plt.xlabel("Algorithm")
plt.ylabel("Accuracy")
